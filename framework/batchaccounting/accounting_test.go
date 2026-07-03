package batchaccounting

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/framework/logstore"
	"github.com/maximhq/bifrost/framework/modelcatalog"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type fakeAccountingStore struct {
	logs map[string]*logstore.Log
	jobs map[string]*logstore.BatchJob
}

func newFakeAccountingStore() *fakeAccountingStore {
	return &fakeAccountingStore{
		logs: make(map[string]*logstore.Log),
		jobs: make(map[string]*logstore.BatchJob),
	}
}

func (s *fakeAccountingStore) CreateIfNotExists(ctx context.Context, entry *logstore.Log) error {
	if _, ok := s.logs[entry.ID]; ok {
		return nil
	}
	copied := *entry
	s.logs[entry.ID] = &copied
	return nil
}

func (s *fakeAccountingStore) UpsertBatchJob(ctx context.Context, job *logstore.BatchJob) error {
	if job.ID == "" {
		job.ID = logstore.BatchJobID(job.Provider, job.BatchID)
	}
	existing, ok := s.jobs[job.ID]
	if !ok {
		copied := *job
		if copied.AccountingStatus == "" {
			copied.AccountingStatus = logstore.BatchJobAccountingStatusPending
		}
		s.jobs[job.ID] = &copied
		return nil
	}
	if job.ProviderStatus != "" {
		existing.ProviderStatus = job.ProviderStatus
	}
	if job.OutputFileID != nil {
		existing.OutputFileID = job.OutputFileID
	}
	return nil
}

func (s *fakeAccountingStore) FindDueBatchJobs(ctx context.Context, provider string, now time.Time, limit int) ([]*logstore.BatchJob, error) {
	var jobs []*logstore.BatchJob
	for _, job := range s.jobs {
		if provider != "" && job.Provider != provider {
			continue
		}
		if job.NextCheckAt == nil || job.NextCheckAt.After(now) {
			continue
		}
		if job.AccountingStatus == logstore.BatchJobAccountingStatusAccounted || job.AccountingStatus == logstore.BatchJobAccountingStatusUnpriceable {
			continue
		}
		jobs = append(jobs, job)
		if limit > 0 && len(jobs) >= limit {
			break
		}
	}
	return jobs, nil
}

func (s *fakeAccountingStore) ClaimBatchJobAccounting(ctx context.Context, jobID string, claimedBy string, ttl time.Duration) (string, bool, error) {
	entry, ok := s.jobs[jobID]
	if !ok {
		return "", false, errors.New("missing batch job")
	}
	if entry.AccountingStatus == logstore.BatchJobAccountingStatusAccounted || entry.AccountingStatus == logstore.BatchJobAccountingStatusUnpriceable {
		return "", false, nil
	}
	if entry.AccountingStatus == logstore.BatchJobAccountingStatusProcessing {
		return "", false, nil
	}
	token := "claim-token-" + jobID
	entry.AccountingStatus = logstore.BatchJobAccountingStatusProcessing
	entry.ClaimedBy = &claimedBy
	expires := time.Now().Add(ttl)
	entry.ClaimExpiresAt = &expires
	entry.ClaimToken = &token
	return token, true, nil
}

func (s *fakeAccountingStore) CompleteBatchJobAccounting(ctx context.Context, id string, claimToken string, logID string, cost float64, promptTokens int, completionTokens int, totalTokens int, modelBreakdown string) error {
	entry, ok := s.jobs[id]
	if !ok {
		return errors.New("missing batch job")
	}
	if entry.ClaimToken == nil || *entry.ClaimToken != claimToken {
		return errors.New("stale claim token")
	}
	entry.AccountingStatus = logstore.BatchJobAccountingStatusAccounted
	entry.AccountedLogID = &logID
	entry.Cost = &cost
	entry.PromptTokens = promptTokens
	entry.CompletionTokens = completionTokens
	entry.TotalTokens = totalTokens
	entry.ModelBreakdown = modelBreakdown
	entry.ClaimExpiresAt = nil
	entry.ClaimToken = nil
	return nil
}

func (s *fakeAccountingStore) MarkBatchJobUnpriceable(ctx context.Context, id string, claimToken string, reason string, err error) error {
	entry, ok := s.jobs[id]
	if !ok {
		return errors.New("missing batch job")
	}
	if entry.ClaimToken == nil || *entry.ClaimToken != claimToken {
		return errors.New("stale claim token")
	}
	entry.AccountingStatus = logstore.BatchJobAccountingStatusUnpriceable
	entry.UnpriceableReason = &reason
	entry.ClaimToken = nil
	entry.ClaimExpiresAt = nil
	return nil
}

func (s *fakeAccountingStore) FailBatchJobAccounting(ctx context.Context, id string, claimToken string, err error) error {
	entry, ok := s.jobs[id]
	if !ok {
		return errors.New("missing batch job")
	}
	entry.AccountingStatus = logstore.BatchJobAccountingStatusError
	entry.ClaimToken = nil
	return nil
}

type fakeBatchPricing struct{}

func (fakeBatchPricing) CalculateCostForUsage(usage *schemas.BifrostLLMUsage, provider schemas.ModelProvider, model string, requestType schemas.RequestType, scopes *modelcatalog.PricingLookupScopes) float64 {
	if usage == nil || provider != schemas.OpenAI || requestType != schemas.BatchResultsRequest {
		return 0
	}
	if usage.Cost != nil && usage.Cost.TotalCost > 0 {
		return usage.Cost.TotalCost
	}
	switch model {
	case "gpt-4o-mini":
		return float64(usage.PromptTokens)*0.000005 + float64(usage.CompletionTokens)*0.000010
	case "gpt-4o":
		return float64(usage.PromptTokens)*0.00001 + float64(usage.CompletionTokens)*0.00002
	default:
		return 0
	}
}

type fakeAggregateLogWriter struct {
	logs map[string]*logstore.Log
}

func (w *fakeAggregateLogWriter) CreateBatchAggregateLog(ctx context.Context, entry *logstore.Log) error {
	if w.logs == nil {
		w.logs = map[string]*logstore.Log{}
	}
	copied := *entry
	w.logs[entry.ID] = &copied
	return nil
}

type fakeUsageReporter struct {
	reports []BatchUsageReport
}

func (r *fakeUsageReporter) ReportBatchUsage(ctx context.Context, report BatchUsageReport) error {
	r.reports = append(r.reports, report)
	return nil
}

func TestAccountBatchResults_OpenAIAggregatesAndWritesOnce(t *testing.T) {
	store := newFakeAccountingStore()
	baseLog := &logstore.Log{
		ID:              "request-1",
		Provider:        string(schemas.OpenAI),
		Model:           "gpt-4o-mini",
		SelectedKeyID:   "key-1",
		SelectedKeyName: "primary",
	}

	req := Request{
		Provider:      schemas.OpenAI,
		BatchID:       "batch_123",
		FallbackModel: "gpt-4o-mini",
		BaseLog:       baseLog,
		ClaimedBy:     "test-node",
		Now:           time.Date(2026, 7, 3, 12, 0, 0, 0, time.UTC),
		Results: []schemas.BatchResultItem{
			openAIResult(200, "gpt-4o-mini", 18, 9),
			openAIResult(200, "gpt-4o", 20, 5),
			openAIResult(500, "gpt-4o-mini", 100, 100),
		},
	}

	summary, err := AccountBatchResults(context.Background(), store, fakeBatchPricing{}, req)
	require.NoError(t, err)
	require.NotNil(t, summary)
	assert.True(t, summary.Accounted)
	assert.Equal(t, 38, summary.Usage.PromptTokens)
	assert.Equal(t, 14, summary.Usage.CompletionTokens)
	assert.Equal(t, 52, summary.Usage.TotalTokens)
	assert.InDelta(t, 0.00048, summary.Cost, 1e-12)
	require.Len(t, store.logs, 1)

	logEntry := store.logs[AccountingLogID(schemas.OpenAI, "batch_123")]
	require.NotNil(t, logEntry)
	assert.Equal(t, "request-1", *logEntry.ParentRequestID)
	assert.Equal(t, string(schemas.BatchResultsRequest), logEntry.Object)
	assert.Equal(t, "mixed", logEntry.Model)
	assert.Equal(t, summary.Cost, *logEntry.Cost)
	assert.Equal(t, "key-1", logEntry.SelectedKeyID)
	assert.Equal(t, true, logEntry.MetadataParsed["batch_accounting"])

	second, err := AccountBatchResults(context.Background(), store, fakeBatchPricing{}, req)
	require.NoError(t, err)
	require.NotNil(t, second)
	assert.False(t, second.Accounted)
	assert.Len(t, store.logs, 1)
}

func TestAccountBatchResults_MissingModelMarksUnpriceable(t *testing.T) {
	store := newFakeAccountingStore()
	result := openAIResult(200, "", 18, 9)

	summary, err := AccountBatchResults(context.Background(), store, fakeBatchPricing{}, Request{
		Provider:  schemas.OpenAI,
		BatchID:   "batch_missing_model",
		ClaimedBy: "test-node",
		Results:   []schemas.BatchResultItem{result},
	})
	require.NoError(t, err)
	require.NotNil(t, summary)
	assert.False(t, summary.Accounted)
	assert.Equal(t, UnpriceableReasonMissingModel, summary.UnpriceableReason)

	job := store.jobs[logstore.BatchJobID(string(schemas.OpenAI), "batch_missing_model")]
	require.NotNil(t, job)
	assert.Equal(t, logstore.BatchJobAccountingStatusUnpriceable, job.AccountingStatus)
	require.NotNil(t, job.UnpriceableReason)
	assert.Equal(t, UnpriceableReasonMissingModel, *job.UnpriceableReason)
	assert.Empty(t, store.logs)
}

func TestAccountBatchResults_MissingBatchPricingMarksUnpriceable(t *testing.T) {
	store := newFakeAccountingStore()

	summary, err := AccountBatchResults(context.Background(), store, fakeBatchPricing{}, Request{
		Provider: schemas.OpenAI,
		BatchID:  "batch_missing_pricing",
		Results:  []schemas.BatchResultItem{openAIResult(200, "unknown-model", 18, 9)},
	})
	require.NoError(t, err)
	require.NotNil(t, summary)
	assert.False(t, summary.Accounted)
	assert.Equal(t, UnpriceableReasonMissingBatchPricing, summary.UnpriceableReason)

	job := store.jobs[logstore.BatchJobID(string(schemas.OpenAI), "batch_missing_pricing")]
	require.NotNil(t, job)
	assert.Equal(t, logstore.BatchJobAccountingStatusUnpriceable, job.AccountingStatus)
	require.NotNil(t, job.UnpriceableReason)
	assert.Equal(t, UnpriceableReasonMissingBatchPricing, *job.UnpriceableReason)
	assert.Empty(t, store.logs)
}

func TestAccountBatchResults_ProviderCostPassthrough(t *testing.T) {
	store := newFakeAccountingStore()
	result := openAIResult(200, "unknown-provider-priced-model", 18, 9)
	result.Response.Body["usage"].(map[string]interface{})["cost"] = map[string]interface{}{
		"total_cost": 0.123,
	}

	summary, err := AccountBatchResults(context.Background(), store, fakeBatchPricing{}, Request{
		Provider:  schemas.OpenAI,
		BatchID:   "batch_provider_cost",
		ClaimedBy: "test-node",
		Results:   []schemas.BatchResultItem{result},
	})
	require.NoError(t, err)
	require.NotNil(t, summary)
	assert.True(t, summary.Accounted)
	assert.InDelta(t, 0.123, summary.Cost, 1e-12)

	breakdown := summary.ModelBreakdowns["unknown-provider-priced-model"]
	assert.True(t, breakdown.ProviderCostUsed)
	assert.InDelta(t, 0.123, breakdown.ProviderCost, 1e-12)
}

func TestAccountBatchResults_UsesAggregateWriterAndUsageReporter(t *testing.T) {
	store := newFakeAccountingStore()
	writer := &fakeAggregateLogWriter{}
	reporter := &fakeUsageReporter{}

	summary, err := AccountBatchResults(context.Background(), store, fakeBatchPricing{}, Request{
		Provider:      schemas.OpenAI,
		BatchID:       "batch_writer",
		FallbackModel: "gpt-4o-mini",
		LogWriter:     writer,
		UsageReporter: reporter,
		Results:       []schemas.BatchResultItem{openAIResult(200, "gpt-4o-mini", 18, 9)},
	})
	require.NoError(t, err)
	require.NotNil(t, summary)
	assert.True(t, summary.Accounted)
	assert.Empty(t, store.logs)
	require.Len(t, writer.logs, 1)
	require.Len(t, reporter.reports, 1)
	assert.Equal(t, AccountingLogID(schemas.OpenAI, "batch_writer"), reporter.reports[0].RequestID)
	assert.Equal(t, int64(27), reporter.reports[0].TokensUsed)
}

type fakeBatchResultFetcher struct {
	retrieveCalls int
	resultsCalls  int
	retrieveResp  *schemas.BifrostBatchRetrieveResponse
	resultsResp   *schemas.BifrostBatchResultsResponse
}

func (f *fakeBatchResultFetcher) RetrieveBatch(ctx context.Context, job *logstore.BatchJob) (*schemas.BifrostBatchRetrieveResponse, error) {
	f.retrieveCalls++
	return f.retrieveResp, nil
}

func (f *fakeBatchResultFetcher) FetchBatchResults(ctx context.Context, job *logstore.BatchJob) (*schemas.BifrostBatchResultsResponse, error) {
	f.resultsCalls++
	return f.resultsResp, nil
}

type fakeKVStore struct {
	setNXAllowed bool
	setNXCalls   int
}

func (s *fakeKVStore) Get(key string) (any, error) {
	return nil, nil
}

func (s *fakeKVStore) SetWithTTL(key string, value any, ttl time.Duration) error {
	return nil
}

func (s *fakeKVStore) SetNXWithTTL(key string, value any, ttl time.Duration) (bool, error) {
	s.setNXCalls++
	return s.setNXAllowed, nil
}

func (s *fakeKVStore) Delete(key string) (bool, error) {
	return true, nil
}

func TestSweeper_AccountsCompletedOpenAIJob(t *testing.T) {
	store := newFakeAccountingStore()
	now := time.Now().UTC().Add(-time.Minute)
	job := &logstore.BatchJob{
		ID:               logstore.BatchJobID(string(schemas.OpenAI), "batch_sweep"),
		Provider:         string(schemas.OpenAI),
		BatchID:          "batch_sweep",
		Model:            "gpt-4o-mini",
		AccountingStatus: logstore.BatchJobAccountingStatusPending,
		NextCheckAt:      &now,
	}
	require.NoError(t, store.UpsertBatchJob(context.Background(), job))

	fetcher := &fakeBatchResultFetcher{
		retrieveResp: &schemas.BifrostBatchRetrieveResponse{
			ID:     "batch_sweep",
			Status: schemas.BatchStatusCompleted,
		},
		resultsResp: &schemas.BifrostBatchResultsResponse{
			BatchID: "batch_sweep",
			Results: []schemas.BatchResultItem{
				openAIResult(200, "gpt-4o-mini", 18, 9),
			},
		},
	}
	sweeper := NewSweeper(store, fakeBatchPricing{}, fetcher, nil, nil, SweeperConfig{
		Provider: schemas.OpenAI,
		Limit:    10,
	})

	sweeper.SweepOnce(context.Background())

	assert.Equal(t, 1, fetcher.retrieveCalls)
	assert.Equal(t, 1, fetcher.resultsCalls)
	accounted := store.jobs[logstore.BatchJobID(string(schemas.OpenAI), "batch_sweep")]
	require.NotNil(t, accounted)
	assert.Equal(t, logstore.BatchJobAccountingStatusAccounted, accounted.AccountingStatus)
	assert.Len(t, store.logs, 1)
}

func TestSweeper_SkipsProviderPollWhenKVLeaseIsHeld(t *testing.T) {
	store := newFakeAccountingStore()
	now := time.Now().UTC().Add(-time.Minute)
	job := &logstore.BatchJob{
		ID:               logstore.BatchJobID(string(schemas.OpenAI), "batch_sweep_lease"),
		Provider:         string(schemas.OpenAI),
		BatchID:          "batch_sweep_lease",
		Model:            "gpt-4o-mini",
		AccountingStatus: logstore.BatchJobAccountingStatusPending,
		NextCheckAt:      &now,
	}
	require.NoError(t, store.UpsertBatchJob(context.Background(), job))

	fetcher := &fakeBatchResultFetcher{}
	kv := &fakeKVStore{setNXAllowed: false}
	sweeper := NewSweeper(store, fakeBatchPricing{}, fetcher, nil, nil, SweeperConfig{
		Provider: schemas.OpenAI,
		Limit:    10,
		KVStore:  kv,
	})

	sweeper.SweepOnce(context.Background())

	assert.Equal(t, 1, kv.setNXCalls)
	assert.Zero(t, fetcher.retrieveCalls)
	assert.Zero(t, fetcher.resultsCalls)
}

func openAIResult(status int, model string, promptTokens int, completionTokens int) schemas.BatchResultItem {
	return schemas.BatchResultItem{
		CustomID: "custom-id",
		Response: &schemas.BatchResultResponse{
			StatusCode: status,
			Body: map[string]interface{}{
				"model": model,
				"usage": map[string]interface{}{
					"prompt_tokens":     promptTokens,
					"completion_tokens": completionTokens,
					"total_tokens":      promptTokens + completionTokens,
				},
			},
		},
	}
}
