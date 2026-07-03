package logstore

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBatchJobAccountingClaimTokenGuardsCompletion(t *testing.T) {
	ctx := context.Background()
	store, err := newSqliteLogStore(ctx, &SQLiteConfig{Path: filepath.Join(t.TempDir(), "batch-jobs.db")}, hybridTestLogger{})
	require.NoError(t, err)

	job := &BatchJob{
		Provider:         "openai",
		BatchID:          "batch_claim_race",
		AccountingStatus: BatchJobAccountingStatusPending,
	}
	require.NoError(t, store.UpsertBatchJob(ctx, job))

	firstToken, claimed, err := store.ClaimBatchJobAccounting(ctx, job.ID, "node-a", time.Minute)
	require.NoError(t, err)
	assert.True(t, claimed)
	assert.NotEmpty(t, firstToken)

	secondToken, claimed, err := store.ClaimBatchJobAccounting(ctx, job.ID, "node-b", time.Minute)
	require.NoError(t, err)
	assert.False(t, claimed)
	assert.Empty(t, secondToken)

	err = store.CompleteBatchJobAccounting(ctx, job.ID, "wrong-token", "log-1", 1.25, 10, 5, 15, "{}")
	assert.True(t, errors.Is(err, ErrNotFound))

	require.NoError(t, store.CompleteBatchJobAccounting(ctx, job.ID, firstToken, "log-1", 1.25, 10, 5, 15, "{}"))

	err = store.CompleteBatchJobAccounting(ctx, job.ID, firstToken, "log-2", 2.50, 20, 10, 30, "{}")
	assert.True(t, errors.Is(err, ErrNotFound))
}
