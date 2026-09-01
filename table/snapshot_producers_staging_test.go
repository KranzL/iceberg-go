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
	"bytes"
	"strings"
	"sync"
	"testing"

	"github.com/apache/iceberg-go"
	iceio "github.com/apache/iceberg-go/io"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestReplaceConcurrencyWriteOptions(t *testing.T) {
	var cfg dataFileCfg
	for _, opt := range []WriteOption{
		WithReplaceValidationConcurrency(3),
		WithReplaceDeletedEntryCollectionConcurrency(4),
		WithReplaceManifestStagingConcurrency(5),
	} {
		opt(&cfg)
	}
	assert.Equal(t, 3, cfg.validationConcurrency)
	assert.Equal(t, 4, cfg.deletedEntryCollectionConcurrency)
	assert.Equal(t, 5, cfg.manifestStagingConcurrency)

	for _, opt := range []WriteOption{
		WithReplaceValidationConcurrency(0),
		WithReplaceDeletedEntryCollectionConcurrency(-1),
		WithReplaceManifestStagingConcurrency(0),
	} {
		opt(&cfg)
	}
	assert.Equal(t, 3, cfg.validationConcurrency, "non-positive values keep the previous setting")
	assert.Equal(t, 4, cfg.deletedEntryCollectionConcurrency)
	assert.Equal(t, 5, cfg.manifestStagingConcurrency)
}

type recordingWriteIO struct {
	iceio.WriteFileIO
	mu      sync.Mutex
	created []string
	removed []string
}

func (r *recordingWriteIO) Create(name string) (iceio.FileWriter, error) {
	r.mu.Lock()
	r.created = append(r.created, name)
	r.mu.Unlock()

	return r.WriteFileIO.Create(name)
}

func (r *recordingWriteIO) Remove(name string) error {
	r.mu.Lock()
	r.removed = append(r.removed, name)
	r.mu.Unlock()

	return r.WriteFileIO.Remove(name)
}

// TestExistingManifestsAbortCleansStagedRewrites pins the failure-path
// contract of the parallel staging pool: when a later manifest fails to read,
// a rewritten manifest already staged by an earlier work item is removed
// instead of leaking as an orphaned object.
func TestExistingManifestsAbortCleansStagedRewrites(t *testing.T) {
	spec := iceberg.NewPartitionSpec()
	txn, wfs := createTestTransactionWithMemIO(t, spec)
	rio := &recordingWriteIO{WriteFileIO: wfs}
	sp := newOverwriteFilesProducer(OpOverwrite, txn, rio, nil, nil)
	sp.producerImpl.(*overwriteFiles).manifestStagingConcurrency = 1

	snapshotID := int64(41)
	removedPath := "mem://default/table-location/data/removed.parquet"
	keptPath := "mem://default/table-location/data/kept.parquet"
	removedDF := newTestDataFile(t, spec, removedPath, nil)
	keptDF := newTestDataFile(t, spec, keptPath, nil)

	var buf bytes.Buffer
	wr, err := iceberg.NewManifestWriter(2, &buf, spec, simpleSchema(), snapshotID)
	require.NoError(t, err)
	require.NoError(t, wr.Add(iceberg.NewManifestEntry(iceberg.EntryStatusADDED, &snapshotID, nil, nil, removedDF)))
	require.NoError(t, wr.Add(iceberg.NewManifestEntry(iceberg.EntryStatusADDED, &snapshotID, nil, nil, keptDF)))
	require.NoError(t, wr.Close())

	goodPath := "mem://default/table-location/metadata/staging-good.avro"
	goodMF, err := wr.ToManifestFile(goodPath, int64(buf.Len()))
	require.NoError(t, err)
	out, err := wfs.Create(goodPath)
	require.NoError(t, err)
	_, err = out.Write(buf.Bytes())
	require.NoError(t, err)
	require.NoError(t, out.Close())

	// A descriptor whose backing file is never written: reading it fails.
	var missingBuf bytes.Buffer
	missingWr, err := iceberg.NewManifestWriter(2, &missingBuf, spec, simpleSchema(), snapshotID)
	require.NoError(t, err)
	require.NoError(t, missingWr.Add(iceberg.NewManifestEntry(iceberg.EntryStatusADDED, &snapshotID, nil, nil,
		newTestDataFile(t, spec, "mem://default/table-location/data/phantom.parquet", nil))))
	require.NoError(t, missingWr.Close())
	missingMF, err := missingWr.ToManifestFile(
		"mem://default/table-location/metadata/staging-missing.avro", int64(missingBuf.Len()))
	require.NoError(t, err)

	listPath := "mem://default/table-location/metadata/staging-list.avro"
	lout, err := wfs.Create(listPath)
	require.NoError(t, err)
	seq := int64(1)
	require.NoError(t, iceberg.WriteManifestList(2, lout, snapshotID, nil, &seq,
		0, []iceberg.ManifestFile{goodMF, missingMF}))
	require.NoError(t, lout.Close())

	parent := &Snapshot{
		SnapshotID:     snapshotID,
		SequenceNumber: seq,
		ManifestList:   listPath,
		Summary:        &Summary{Operation: OpAppend},
	}

	sp.deleteDataFile(removedDF)

	_, err = sp.existingManifests(t.Context(), parent)
	require.Error(t, err)

	rio.mu.Lock()
	defer rio.mu.Unlock()
	stagedRewrites := make([]string, 0, len(rio.created))
	for _, path := range rio.created {
		if strings.Contains(path, "-m") && strings.HasSuffix(path, ".avro") {
			stagedRewrites = append(stagedRewrites, path)
		}
	}
	require.NotEmpty(t, stagedRewrites,
		"the good manifest carries a removed file, so staging must have rewritten it before the abort")
	for _, path := range stagedRewrites {
		assert.Contains(t, rio.removed, path,
			"aborted staging must remove the rewritten manifest it wrote")
	}
}

// TestExistingManifestsParallelMatchesManifestListOrder pins that the staged
// output preserves manifest-list order even when several workers stage
// manifests concurrently.
func TestExistingManifestsParallelMatchesManifestListOrder(t *testing.T) {
	spec := iceberg.NewPartitionSpec()
	txn, wfs := createTestTransactionWithMemIO(t, spec)
	sp := newOverwriteFilesProducer(OpOverwrite, txn, wfs, nil, nil)
	sp.producerImpl.(*overwriteFiles).manifestStagingConcurrency = 4

	snapshotID := int64(42)
	const numManifests = 6
	manifests := make([]iceberg.ManifestFile, numManifests)
	for i := range numManifests {
		path := "mem://default/table-location/metadata/order-" + string(rune('a'+i)) + ".avro"
		df := newTestDataFile(t, spec,
			"mem://default/table-location/data/order-"+string(rune('a'+i))+".parquet", nil)

		var buf bytes.Buffer
		wr, err := iceberg.NewManifestWriter(2, &buf, spec, simpleSchema(), snapshotID)
		require.NoError(t, err)
		require.NoError(t, wr.Add(iceberg.NewManifestEntry(iceberg.EntryStatusADDED, &snapshotID, nil, nil, df)))
		require.NoError(t, wr.Close())
		manifests[i], err = wr.ToManifestFile(path, int64(buf.Len()))
		require.NoError(t, err)

		out, err := wfs.Create(path)
		require.NoError(t, err)
		_, err = out.Write(buf.Bytes())
		require.NoError(t, err)
		require.NoError(t, out.Close())
	}

	listPath := "mem://default/table-location/metadata/order-list.avro"
	lout, err := wfs.Create(listPath)
	require.NoError(t, err)
	seq := int64(1)
	require.NoError(t, iceberg.WriteManifestList(2, lout, snapshotID, nil, &seq, 0, manifests))
	require.NoError(t, lout.Close())

	parent := &Snapshot{
		SnapshotID:     snapshotID,
		SequenceNumber: seq,
		ManifestList:   listPath,
		Summary:        &Summary{Operation: OpAppend},
	}

	got, err := sp.existingManifests(t.Context(), parent)
	require.NoError(t, err)
	require.Len(t, got, numManifests)
	for i, m := range manifests {
		assert.Equal(t, m.FilePath(), got[i].FilePath())
	}
}
