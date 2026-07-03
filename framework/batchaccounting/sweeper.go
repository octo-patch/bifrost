package batchaccounting

import (
	"context"
	"fmt"
	"time"

	"github.com/bytedance/sonic"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/framework/logstore"
	"github.com/maximhq/bifrost/framework/modelcatalog"
)

const (
	defaultSweepInterval = time.Minute
	defaultSweepLimit    = 50
)

type SweepStore interface {
	Store
	FindDueBatchJobs(ctx context.Context, provider string, now time.Time, limit int) ([]*logstore.BatchJob, error)
}

type BatchResultFetcher interface {
	RetrieveBatch(ctx context.Context, job *logstore.BatchJob) (*schemas.BifrostBatchRetrieveResponse, error)
	FetchBatchResults(ctx context.Context, job *logstore.BatchJob) (*schemas.BifrostBatchResultsResponse, error)
}

type SweeperConfig struct {
	Interval   time.Duration
	Limit      int
	ClaimedBy  string
	Provider   schemas.ModelProvider
	Scopes     *modelcatalog.PricingLookupScopes
	KVStore    schemas.KVStore
	KVLeaseTTL time.Duration
}

type Sweeper struct {
	store         SweepStore
	pricing       PricingManager
	fetcher       BatchResultFetcher
	logWriter     AggregateLogWriter
	usageReporter UsageReporter
	config        SweeperConfig
}

func NewSweeper(store SweepStore, pricing PricingManager, fetcher BatchResultFetcher, logWriter AggregateLogWriter, usageReporter UsageReporter, config SweeperConfig) *Sweeper {
	if config.Interval <= 0 {
		config.Interval = defaultSweepInterval
	}
	if config.Limit <= 0 {
		config.Limit = defaultSweepLimit
	}
	if config.Provider == "" {
		config.Provider = schemas.OpenAI
	}
	if config.ClaimedBy == "" {
		config.ClaimedBy = "batch-sweeper"
	}
	if config.KVLeaseTTL <= 0 {
		config.KVLeaseTTL = config.Interval
	}
	return &Sweeper{
		store:         store,
		pricing:       pricing,
		fetcher:       fetcher,
		logWriter:     logWriter,
		usageReporter: usageReporter,
		config:        config,
	}
}

func (s *Sweeper) Run(ctx context.Context) {
	if s == nil {
		return
	}
	ticker := time.NewTicker(s.config.Interval)
	defer ticker.Stop()
	for {
		s.SweepOnce(ctx)
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

func (s *Sweeper) SweepOnce(ctx context.Context) {
	if s == nil || s.store == nil || s.pricing == nil || s.fetcher == nil {
		return
	}
	now := time.Now().UTC()
	jobs, err := s.store.FindDueBatchJobs(ctx, string(s.config.Provider), now, s.config.Limit)
	if err != nil {
		return
	}
	for _, job := range jobs {
		s.sweepJob(ctx, job, now)
	}
}

func (s *Sweeper) sweepJob(ctx context.Context, job *logstore.BatchJob, now time.Time) {
	if job == nil || schemas.ModelProvider(job.Provider) != schemas.OpenAI {
		return
	}
	locked, err := s.acquireProviderPollLease(job)
	if err != nil || !locked {
		return
	}
	retrieved, err := s.fetcher.RetrieveBatch(ctx, job)
	if err != nil {
		s.reschedule(ctx, job, now)
		return
	}
	latest := batchJobFromRetrieve(job, retrieved, now)
	if err := s.store.UpsertBatchJob(ctx, latest); err != nil {
		return
	}
	if retrieved.Status != schemas.BatchStatusCompleted {
		if isTerminalStatus(retrieved.Status) {
			s.markTerminalWithoutResults(ctx, latest)
			return
		}
		s.reschedule(ctx, latest, now)
		return
	}

	results, err := s.fetcher.FetchBatchResults(ctx, latest)
	if err != nil || results == nil {
		s.reschedule(ctx, latest, now)
		return
	}
	_, _ = AccountBatchResults(ctx, s.store, s.pricing, Request{
		Provider:      schemas.ModelProvider(latest.Provider),
		BatchID:       latest.BatchID,
		FallbackModel: latest.Model,
		Results:       results.Results,
		BatchJob:      latest,
		LogWriter:     s.logWriter,
		UsageReporter: s.usageReporter,
		ClaimedBy:     s.config.ClaimedBy,
		Scopes:        s.config.Scopes,
		Now:           now,
	})
}

func (s *Sweeper) reschedule(ctx context.Context, job *logstore.BatchJob, now time.Time) {
	next := now.Add(s.config.Interval)
	job.NextCheckAt = &next
	job.LastCheckedAt = &now
	_ = s.store.UpsertBatchJob(ctx, job)
}

func (s *Sweeper) markTerminalWithoutResults(ctx context.Context, job *logstore.BatchJob) {
	token, claimed, err := s.store.ClaimBatchJobAccounting(ctx, job.ID, s.config.ClaimedBy, defaultClaimTTL)
	if err != nil || !claimed {
		return
	}
	_ = s.store.MarkBatchJobUnpriceable(ctx, job.ID, token, "terminal_without_results", nil)
}

func batchJobFromRetrieve(existing *logstore.BatchJob, retrieved *schemas.BifrostBatchRetrieveResponse, now time.Time) *logstore.BatchJob {
	job := *existing
	job.BatchID = retrieved.ID
	job.ProviderStatus = string(retrieved.Status)
	job.Endpoint = retrieved.Endpoint
	job.InputFileID = retrieved.InputFileID
	job.OutputFileID = retrieved.OutputFileID
	job.ErrorFileID = retrieved.ErrorFileID
	job.ResultsURL = retrieved.ResultsURL
	job.OperationName = retrieved.OperationName
	job.RequestCounts = marshalString(retrieved.RequestCounts)
	job.LastCheckedAt = &now
	if !isTerminalStatus(retrieved.Status) {
		next := now.Add(defaultSweepInterval)
		job.NextCheckAt = &next
	}
	return &job
}

func marshalString(value any) string {
	out, err := sonic.MarshalString(value)
	if err != nil || out == "{}" {
		return ""
	}
	return out
}

func isTerminalStatus(status schemas.BatchStatus) bool {
	switch status {
	case schemas.BatchStatusCompleted, schemas.BatchStatusFailed, schemas.BatchStatusExpired, schemas.BatchStatusCancelled, schemas.BatchStatusEnded, schemas.BatchStatusDeleted:
		return true
	default:
		return false
	}
}

func (s *Sweeper) acquireProviderPollLease(job *logstore.BatchJob) (bool, error) {
	if s.config.KVStore == nil {
		return true, nil
	}
	key := fmt.Sprintf("batch-accounting:poll:%s:%s", job.Provider, job.BatchID)
	value := map[string]string{
		"claimed_by": s.config.ClaimedBy,
		"job_id":     job.ID,
	}
	return s.config.KVStore.SetNXWithTTL(key, value, s.config.KVLeaseTTL)
}

func BatchFetcherError(provider schemas.ModelProvider, batchID string, err error) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("batch fetch failed for provider=%s batch_id=%s: %w", provider, batchID, err)
}
