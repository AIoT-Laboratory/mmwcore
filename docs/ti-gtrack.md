# Complete TI GTRACK for IWR6843

`TiGTrack3D` executes the original **3DA nine-state** `gtrack_create / gtrack_step / gtrack_delete`
from Radar Toolbox **4.00.00.05**, custom SDK3 `trackerproc_overhead`. This is the source selected
by that release's 6843 3D People Tracking project. The earlier `GTrack3D` six-state Rust tracker
remains available for historical comparisons; its results are not stock TI results.

This implements the complete tracking layer. The application still supplies its existing RPC
measurements. Capon/angle processing, point detection and TI board execution are separate work;
this host backend does not establish full People Tracking demo or board-binary equivalence.

## Source and build boundary

The [source lock](../tools/ti_gtrack/source-lock.json) pins all 23 required TI C/header files.
[build.py](../tools/ti_gtrack/build.py) reads an external SDK installation, rejects changed sources,
and writes a local library, manifest and `TI-LICENSE.txt`. TI algorithm sources are unmodified.
No TI algorithm source or binary is included in the Apache Python/Rust distribution. The local
plugin retains its TI-device-only license. Operational build/start commands are in
[OpenMMW commands](../../openmmw/docs/commands.md#完整-ti-gtrack-本地后端).

The new `mmwcore-ti-gtrack` Rust crate owns the checked native ABI and dynamic library lifetime.
Unsafe calls are confined to that host crate; the existing numerical core and PyO3 crate retain
`forbid(unsafe_code)`. Loading requires an explicitly selected local manifest, matching binary
SHA-256, ABI version and Config/Target structure sizes. A manifest is provenance, not a signature
or a sandbox for arbitrary native code. The supplied SDK build is the intended source.

One host allocator correction is necessary: stock step clears `(N >> 3) + 1` bitmap bytes while
create allocates `ceil(N/8)`. At capacities divisible by 8 it writes one byte beyond each of two
bitmaps. The host allocator reserves one extra zero byte per allocation, with overflow checks;
the TI numerical sources and step order are unchanged.

## Capability coverage

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
with TiGTrack3D(spec, plugin_manifest="build/ti-gtrack/manifest.json") as tracker:
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
10 Hz sparse-RPC pipeline. Its [baseline record](../../openmmw/docs/research/ti-gtrack-rpc-baseline-v1.md)
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

## Validation and current research result

The independent C oracle calls original TI directly without this bridge. Four 99-frame cases
(wall/ceiling × absent/positive variances) matched every common float32 report field exactly in the
local GCC host comparison. They cover motion, static transitions, empty frames, deletion, uid
reuse with new tid, and presence. Committed [oracle outputs](../tests/fixtures/ti_gtrack_3da_oracle.json)
support end-to-end Python regression with a small host portability tolerance. The native bridge
is additionally tested for input rejection, no truncation, coordinate signs, reset/close and
binary-hash mismatch. Numerical agreement is host-source agreement, not TI DSP bit equivalence.

On 850 frozen pilot-002 RPC frames, the complete source with ISK tracking defaults ran without
non-finite states and allocated **zero** tracks. Only 35 frames even contained 20 points globally;
c08 had at most 18. The remaining 35 frames did not pass the complete allocation conditions.
This historical ISK run checks integration and identifies an input/configuration mismatch; it does
not demonstrate human tracking quality or disprove TI capability.
See [bounded replay](../../openmmw/outputs/experiments/gtrack-stock-pilot002-v1/summary.json).
Following the user's request for an application baseline, RPC v1 was frozen before one new replay.
c08 reports one uninterrupted ID on frames 25–99, while c01–c03 still do not allocate and c04–c07
retain extra branches. These are diagnostic counts on previously explored data, not independent
person-tracking accuracy. Simplified wrappers still match the independent 396-frame C oracle;
all 850 real and 1,080 synthetic empty-tail reports reproduce exactly with the final code,
excluding host timing counters.
