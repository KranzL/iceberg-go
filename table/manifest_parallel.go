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
	"iter"

	"github.com/apache/iceberg-go"
	"github.com/apache/iceberg-go/config"
	iceio "github.com/apache/iceberg-go/io"
	"golang.org/x/sync/errgroup"
)

const manifestPoolCancelCheckInterval = 4096

func manifestPoolSize(configured int) int {
	if configured > 0 {
		return configured
	}
	if workers := config.EnvConfig.MaxWorkers; workers > 0 {
		return workers
	}

	return 1
}

func mapManifestsOrdered[T any](ctx context.Context, concurrency int, manifests []iceberg.ManifestFile,
	fn func(context.Context, iceberg.ManifestFile) (T, error),
) ([]T, error) {
	results := make([]T, len(manifests))
	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(manifestPoolSize(concurrency))
	for i, m := range manifests {
		g.Go(func() error {
			if err := context.Cause(gctx); err != nil {
				return err
			}

			out, err := fn(gctx, m)
			if err != nil {
				return err
			}
			results[i] = out

			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return nil, err
	}

	return results, nil
}

func manifestEntriesWithCancel(ctx context.Context, m iceberg.ManifestFile, fio iceio.IO, discardDeleted bool) iter.Seq2[iceberg.ManifestEntry, error] {
	return func(yield func(iceberg.ManifestEntry, error) bool) {
		n := 0
		for entry, err := range m.Entries(fio, discardDeleted) {
			if err == nil && n%manifestPoolCancelCheckInterval == 0 {
				if ctxErr := context.Cause(ctx); ctxErr != nil {
					yield(nil, ctxErr)

					return
				}
			}
			n++
			if !yield(entry, err) {
				return
			}
		}
	}
}
