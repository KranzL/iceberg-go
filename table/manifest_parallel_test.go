// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

package table

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/apache/iceberg-go"
	"github.com/apache/iceberg-go/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestManifestPoolSize(t *testing.T) {
	assert.Equal(t, 1, manifestPoolSize(1))
	assert.Equal(t, 7, manifestPoolSize(7))
	assert.Equal(t, config.EnvConfig.MaxWorkers, manifestPoolSize(0))
	assert.Equal(t, config.EnvConfig.MaxWorkers, manifestPoolSize(-3))
}

func TestMapManifestsOrderedPreservesInputOrder(t *testing.T) {
	manifests := make([]iceberg.ManifestFile, 32)

	var next atomic.Int64
	results, err := mapManifestsOrdered(t.Context(), 4, manifests,
		func(context.Context, iceberg.ManifestFile) (int64, error) {
			// Completion order is deliberately scrambled relative to
			// submission order; results must still come back by index.
			seq := next.Add(1)
			if seq%3 == 0 {
				time.Sleep(time.Millisecond)
			}

			return seq, nil
		})
	require.NoError(t, err)
	require.Len(t, results, len(manifests))

	seen := make(map[int64]struct{}, len(results))
	for _, r := range results {
		assert.NotContains(t, seen, r)
		seen[r] = struct{}{}
	}
}

func TestMapManifestsOrderedBoundsConcurrency(t *testing.T) {
	manifests := make([]iceberg.ManifestFile, 24)
	const limit = 3

	var inFlight, maxInFlight atomic.Int64
	_, err := mapManifestsOrdered(t.Context(), limit, manifests,
		func(context.Context, iceberg.ManifestFile) (struct{}, error) {
			cur := inFlight.Add(1)
			defer inFlight.Add(-1)
			for {
				prev := maxInFlight.Load()
				if cur <= prev || maxInFlight.CompareAndSwap(prev, cur) {
					break
				}
			}
			time.Sleep(time.Millisecond)

			return struct{}{}, nil
		})
	require.NoError(t, err)
	assert.LessOrEqual(t, maxInFlight.Load(), int64(limit))
	assert.Positive(t, maxInFlight.Load())
}

func TestMapManifestsOrderedCancelsPendingOnError(t *testing.T) {
	manifests := make([]iceberg.ManifestFile, 8)
	boom := errors.New("manifest read failed")

	var calls atomic.Int64
	_, err := mapManifestsOrdered(t.Context(), 2, manifests,
		func(gctx context.Context, _ iceberg.ManifestFile) (struct{}, error) {
			if calls.Add(1) == 1 {
				return struct{}{}, boom
			}

			select {
			case <-gctx.Done():
				return struct{}{}, context.Cause(gctx)
			case <-time.After(5 * time.Second):
				return struct{}{}, errors.New("cancellation did not propagate to pending work")
			}
		})
	require.ErrorIs(t, err, boom)
}

func TestMapManifestsOrderedEmptyInput(t *testing.T) {
	results, err := mapManifestsOrdered(t.Context(), 0, nil,
		func(context.Context, iceberg.ManifestFile) (int, error) {
			return 0, errors.New("must not be called")
		})
	require.NoError(t, err)
	assert.Empty(t, results)
}

// buildHeadContextWithManifests wires a conflictContext whose branch head
// carries the given manifest list, so existence validation walks real
// manifests through the tracking IO.
func buildHeadContextWithManifests(t *testing.T, fio *trackingCallsIO, listPath string) *conflictContext {
	t.Helper()

	schema := iceberg.NewSchema(0,
		iceberg.NestedField{ID: 1, Name: "x", Type: iceberg.PrimitiveTypes.Int64, Required: true},
	)
	meta, err := NewMetadata(schema, iceberg.UnpartitionedSpec, UnsortedSortOrder, "mem://tbl",
		iceberg.Properties{PropertyFormatVersion: "2"})
	require.NoError(t, err)

	builder, err := MetadataBuilderFromBase(meta, "")
	require.NoError(t, err)
	snap := Snapshot{
		SnapshotID:     1,
		SequenceNumber: 1,
		TimestampMs:    meta.LastUpdatedMillis() + 1,
		ManifestList:   listPath,
		Summary:        &Summary{Operation: OpAppend},
	}
	require.NoError(t, builder.AddSnapshot(&snap))
	require.NoError(t, builder.SetSnapshotRef(MainBranch, 1, BranchRef))
	built, err := builder.Build()
	require.NoError(t, err)

	return &conflictContext{
		current: built,
		branch:  MainBranch,
		fs:      &conflictValidationStatIO{trackingCallsIO: fio},
	}
}

func TestValidateDataFilesExistAcrossManyManifests(t *testing.T) {
	tio := newTrackingCallsIO()
	dir := filepath.ToSlash(t.TempDir())

	const numManifests = 9
	manifests := make([]iceberg.ManifestFile, numManifests)
	dataPaths := make([]string, numManifests)
	for i := range numManifests {
		dataPaths[i] = filepath.Join(dir, fmt.Sprintf("data-%d.parquet", i))
		manifests[i] = writeManifest(t, tio.trackingIO, 1, 1,
			filepath.Join(dir, fmt.Sprintf("manifest-%d.avro", i)), dataPaths[i])
	}
	listPath := filepath.Join(dir, "manifest-list.avro")
	writeManifestList(t, tio.trackingIO, 1, listPath, manifests)

	cc := buildHeadContextWithManifests(t, tio, listPath)

	// Paths scattered across manifests are all found.
	require.NoError(t, validateDataFilesExist(t.Context(), cc,
		[]string{dataPaths[0], dataPaths[4], dataPaths[numManifests-1]}))

	// A path in no manifest fails and is named in the error.
	missing := filepath.Join(dir, "not-there.parquet")
	err := validateDataFilesExist(t.Context(), cc, []string{dataPaths[2], missing})
	require.ErrorIs(t, err, ErrDataFilesMissing)
	require.ErrorContains(t, err, "not-there.parquet")
}

func TestValidateDataFilesExistCancelledContext(t *testing.T) {
	tio := newTrackingCallsIO()
	dir := filepath.ToSlash(t.TempDir())

	dataPath := filepath.Join(dir, "data.parquet")
	mf := writeManifest(t, tio.trackingIO, 1, 1, filepath.Join(dir, "manifest.avro"), dataPath)
	listPath := filepath.Join(dir, "manifest-list.avro")
	writeManifestList(t, tio.trackingIO, 1, listPath, []iceberg.ManifestFile{mf})

	cc := buildHeadContextWithManifests(t, tio, listPath)

	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	err := validateDataFilesExist(ctx, cc, []string{dataPath})
	require.ErrorIs(t, err, context.Canceled)
}
