# Complete TI GTRACK for IWR6843

`TiGTrack3D` runs a safe Rust adaptation of the **3DA nine-state** tracker from
Radar Toolbox **4.00.00.05**, custom SDK3 `trackerproc_overhead`. Multiscale tracking
(`ScatterBodyTracker`) is the retained alternative for comparison; the older simplified
six-state and 2D tracker prototypes have been removed.
The application supplies RPC measurements; this is not a port of the entire board demo.

## Source and build boundary

The default mmwcore wheel includes the tracker. No TI SDK, C compiler, plugin manifest
or DLL is required at installation or runtime. The old `MMWCORE_TI_GTRACK_MANIFEST`
environment variable no longer selects a runtime backend.

The [Rust component](../crates/mmwcore-ti-gtrack/src/lib.rs) separates frame orchestration
(`tracker.rs`), per-track state/update/lifecycle (`unit.rs`), read-only association gates
(`association.rs`) and fixed-size numerical operations (`math.rs`). Internal track and velocity
states are enums, lifecycle counters have named `u16` fields, and each point has one typed
association record. TI numeric codes and parallel report arrays are produced only at the output
boundary; C layout is enabled only for the optional oracle build.

The tracker owns reusable frame buffers. Candidate selection swaps two buffers instead of
allocating for every seed; track updates share one scratch buffer. Static support queries the
same immutable gate without constructing a temporary tracker input or rolling back counters.
These are implementation changes, not new tracking rules. TI's lookup interpolation,
absolute-product group dispersion, threshold comparisons and slot reuse behavior are retained.
Changes to these algorithms need separate validation.
The source [lock](../tools/ti_gtrack/source-lock.json) identifies the C version used for comparison.

This adaptation retains the [TI device-only license](../crates/mmwcore-ti-gtrack/TI-LICENSE.txt).
Source and wheel distributions include that license and NOTICE; the remainder of mmwcore
stays Apache-2.0. Rust adaptation does not remove TI's licensing conditions.

The existing [C build tool](../tools/ti_gtrack/build.py) is only a development oracle.
`cargo test -p mmwcore-ti-gtrack --features reference-plugin` additionally compares against
`build/ti-gtrack/manifest.json`. Normal builds exclude its dynamic loader dependencies.
Python oracle comparison explicitly uses `plugin_manifest=...` and requires a
`maturin build --features ti-reference` development wheel. The ordinary constructor is
`TiGTrack3D(spec)`.

## Capability coverage

Optional `step(point_cloud, static_positions=xyz)` supplies `(M,3)` sensor
forward/right/up positions to support existing tracks during RPC loss.
The built-in implementation includes `static_support_v1`; the ordinary step is unchanged.
The host associates RPC first, then uses the original TI gate for static candidates.
Only a candidate matching exactly one existing ACTIVE track with no RPC association
can support it; both prediction and candidate must be inside the full boundary ROI.
The narrower static box does not reject these supplemental observations. Ambiguous,
unmatched and tentative-track candidates cannot allocate or update a position centroid.
The appended zero-speed rows use unit SNR solely as an inert ABI placeholder.

TI still decides the moving/static transition. Once static, a fresh accepted support
resets the independent sleep counter before the original event logic; otherwise
its integer fine-motion point history can expire a unit even with one static point
every frame. No observation means the original miss/sleep/exit rules apply. This is
an explicit host extension, not claimed as unmodified TI association/lifecycle behavior.
It establishes an integration mechanism, not human attribution or field performance.

`observation_track_ids` retains the RPC prefix. The full native report retains all
rows and adds `static_support` (`rpc_count`, sensor XYZ `positions`, `assigned_count`).
Combined capacity is checked without truncation; explicit mixed RPC variances are
currently rejected. Static candidates are not new RPC measurements or accurate velocity
measurements. The Rust frame pipeline applies this extension after RPC association and before
lifecycle updates. The C oracle retains its linker wrappers for comparison.

| Stage | Maintained source behavior |
|---|---|
| Predict | 9D position/velocity/acceleration CA model, full 9×9 covariance, four-dimensional spherical measurement prediction |
| Associate | Per-point bidding, spatial partial Mahalanobis gate, physical limits, full score with weighted Doppler, ambiguity/unique bitmap, static-point handling |
| Allocate | Original iterative candidate selection, spherical centroid, independent distance/velocity checks, point/SNR/velocity conditions, range-dependent and obscured SNR logic |
| Update | 3D EKF, dynamic unique support, wall-mount SNR weighting, group dispersion, centroid uncertainty, expected point count, measurement variances, velocity unrolling state machine |
| Lifecycle | DETECTION/ACTIVE/FREE rules, reliable-point counters, static/moving transitions, normal/static/exit/sleep deletion, world scenery checks |
| Installation | Native wall/ceiling branches, elevation/azimuth tilt, sensor height, boundary/static/occupancy boxes, presence detection |
| Report | All active units including DETECTION; raw uid/tid, state and covariance, group covariance/dispersion, EC, gain, dimensions, confidence, point labels/unique/static/score, updated Doppler, presence, benchmark ticks |

This pinned 3DA source disables ghost marking. Do not describe ghost suppression from another
GTRACK version as enabled here. See the [source review](research/gtrack-capability-source-review.md)
for version-specific branches and corrections to conceptual GTRACK descriptions.

## Python API and coordinates

```python
from mmwcore.core import Box3D
from mmwcore.tracking import TiGTrack3D, TiGTrack3DSpec, TiGTrackScenery

spec = TiGTrack3DSpec(
    frame_period_s=0.1,
    max_radial_velocity_mps=4.0,       # use the actual capture profile
    radial_velocity_resolution_mps=0.125,
    scenery=TiGTrackScenery(boundary_boxes=(Box3D(0.5, 6, -3, 3, 0, 3),)),
)
with TiGTrack3D(spec) as tracker:
    tracks = tracker.step(point_cloud)
    native_report = tracker.last_report
```

- Cartesian input is `sensor_forward_lateral_up` or `sensor_forward_right_up`: **forward,
  right, up**. It must include radial `velocity` and linear `snr` or `snr_db`.
- `step_spherical(points, variances=None)` takes `(N,5)` range m, right-positive azimuth rad,
  up-positive elevation rad, radial velocity m/s and **linear** SNR. Both methods advance once;
  they are alternative input routes, not two stages to call for the same frame.
- Doppler is **approaching negative, receding positive**, unchanged across both routes.
- Optional variance is `(N,4)` in m²/rad²/rad²/(m/s)². Every explicit entry must be positive
  finite. Use `None` when unknown; zero is not an unknown-noise placeholder.
- `TiGTrackScenery` boxes use world forward/right/up. Maximum accelerations use sensor
  forward/right/up; the adapter swaps axes for TI. Scenery horizontal origin must be the
  radar: this source's transform uses sensor height, not horizontal translation. Positive
  elevation tilt is downward. OpenMMW takes installation and ROI from the capture snapshot.
  At least one boundary box is required: in this pinned source zero boxes count every unit as
  outside and delete it at the exit threshold; zero does not disable the boundary check.
- Input capacity is checked without truncation. Finite forward-hemisphere measurements with
  positive range and SNR are required. Inputs are copied before stock Doppler unrolling.
- `reset()` creates a fresh source instance and restarts its IDs. `close()` releases it.
  Invalid inputs do not advance the tracker. A non-finite native result emits an error and
  requires reset; it is never serialized as a plausible partial/null-valued track.

`TiGTrackGating`, `TiGTrackAllocation`, `TiGTrackLifecycle` and `TiGTrackScenery` expose the source
configuration fields. Defaults follow the pinned **ISK_6m_default.cfg tracking layer**:

| Setting | Default |
|---|---|
| Gate gain; depth/width/height/velocity limits | 3; 2 m / 2 m / 2 m / 4 m/s |
| Allocation SNR / obscured SNR / velocity / points / distance / velocity difference | 40 / 100 / 0.1 m/s / 20 / 0.5 m / 20 m/s |
| det2act / det2free / active2free / static2free / exit2free / sleep2free | 3 / 3 / 12 / 500 / 5 / 6000 |
| Maximum points / tracks; acceleration | 800 / 30; 0.1 m/s² per axis |

These remain **mmwcore API defaults**. OpenMMW now explicitly loads its versioned
[RPC v1 application profile](../../openmmw/openmmw/configs/ti_gtrack_rpc_v1.json) for the local
10 Hz sparse-RPC pipeline. Its [baseline record](../../openmmw/docs/research/baseline.md#ti-gtrack-rpc-baseline-v1)
documents parameter choices, a fixed replay and unresolved cases. It does not change these defaults
or the pinned TI numerical source.

Timing, maximum radial velocity and velocity resolution remain required capture parameters.
Installation/ROI are application inputs, not the example room. Presence is disabled until
occupancy boxes and a positive presence point threshold are configured. Generic library defaults
are different from these application defaults; in particular a zero velocity gate limit produced
a singular/non-finite native result in the synthetic regression. No fallback gate is substituted.

## Reading the report

`TrackFrame` exposes sensor forward/right/up position and velocity, position covariance and an
extent covariance obtained by projecting native **spherical group dispersion** through its
Cartesian Jacobian. This extent is reflection spread, not an anatomical body size. Its metadata
retains the full report, including acceleration. UI applies installation rotation to all displayed
vectors/covariances and displays every confirmed/coasting track.

The raw `targets` use TI **right/forward/up** axes. `sensor_targets` provide the reordered nine-state
view. `uid` is a reusable pool slot; `tid` is the increasing identity. Raw `point_uid` retains stock
labels (0–199 slots; 254 outside/filtered, 255 unassociated; other reserved values remain raw).
`point_tid` maps only to surviving reported units, using -1 otherwise. A slot can still be present
on a point after its track was deleted during Update, so -1 alone is not proof of association failure.

`point_static` preserves module `isStaticIndex`, a **Score-stage association bookkeeping flag**.
It is not the point's zero-Doppler classification or a reliable final target-state label.
Pinned Update defines dynamic points using `abs(doppler) > FLT_EPSILON` after Score's unrolling;
good/reliable points are dynamic and unique. For surviving units, combine membership,
`updated_doppler` and `point_unique` to interpret that criterion, rather than `!point_static`.
Non-unique dynamic points can still support ACTIVE lifecycle hits without entering the good-point
centroid; static-target lifecycle and confidence also have separate point-use rules.

Raw state 2 maps to tentative, state 3 to confirmed/coasting; coasting is the application label
when TI `active2freeCount` is nonzero, not an additional native state. Counter order is
detect2active, detect2free, active2free, sleep2free, outside2free, static-point history.

**EC is the cached inverse group covariance**, not the 9×9 state covariance or necessarily the
inverse of the post-Update group covariance. Stock cache behavior at birth/reused slots is retained.
`apriori_state_after_step` and `apriori_covariance_after_step` are snapshots after the entire step:
Update can overwrite these buffers during static transitions. They are not clean Predict-stage
hooks. `predicted_measurement` retains `H_s`; do not pair it blindly with overwritten apriori
buffers for a future RT query. A proper Predict-stage hook is a separate future change.

## Validation

Default tests run without a TI installation: 396 frozen original-C frames plus the
Python lifecycle, capacity, coordinate and static-support regressions.
The optional C-oracle test adds 1,440 frames of two-target association, support-only
observations, disappearance/rebirth, velocity aliases, explicit variances and 0/30/90-degree mounts.
Float comparisons retain rtol=3e-6, atol=2e-7; IDs and point labels must match exactly.

On this Windows x64 host, replaying c01-c08 (800 actual ADC-to-RPC frames, RPC v1,
recorded one-person limit) matched the C backend's point labels exactly. Maximum
absolute difference in the nine-state vectors and state covariances was zero.
This establishes migration parity on these inputs, not independent tracking accuracy
or TI DSP bit equivalence. No training or field hardware run was performed.
