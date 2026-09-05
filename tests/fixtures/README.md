# Complete TI oracle fixture

`ti_gtrack_3da_oracle.json` contains numeric outputs from an independent C executable calling
the external pinned TI `gtrack_create` and `gtrack_step`, not the mmwcore bridge. It contains no
TI numerical source. Source version/hashes are in `tools/ti_gtrack/source-lock.json`.

The input construction and complete configuration are in `test_mmwcore_ti_gtrack.py`: each case
has 6 moving, 12 static, 60 empty, 6 new moving and 15 empty frames. Cases are tilt 0/90 degrees
and absent/present positive measurement variance. SNR/allocation values are synthetic branch
coverage settings, not the application's ISK defaults or local pilot tuning.

The fixture stores flattened native matrices and nine-significant-digit float outputs. Tests
reshape before comparison. Provenance records independent oracle and bridge source hashes;
the local review artifacts (including oracle.c and raw input/output trace) are retained under
`build/ti-gtrack-review`. Exact float32 agreement was checked there; portable regression uses
rtol 3e-6 / atol 2e-7 for host math-library variation, with exact IDs, flags, presence and counts.
