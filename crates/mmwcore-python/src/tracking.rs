//! PyO3 boundary for stateful native target tracking.

use std::sync::Mutex;

use mmwcore::{
    Box2D, Box3D, ClusterMeasurements, ClusterTracker2D, GTrack3D, GTrack3DDiagnostics,
    PointMeasurements, PointTracker2D, TrackAllocationConfig, TrackGatingConfig,
    TrackLifecycleConfig, TrackScenery3DConfig, TrackSceneryConfig, TrackStepResult,
    Tracker2DConfig, Tracker3DConfig, TrackerDynamics3DConfig, TrackerDynamicsConfig,
    TrackingError, TrackingMetricsError, TrackingMetricsInput, TrackingSequenceMetrics,
    summarize_tracking_metrics as native_summarize_tracking_metrics,
};
use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::{NativeDbscanConfig, dbscan_config};

#[pyclass]
struct NativeClusterTracker2D {
    tracker: Mutex<ClusterTracker2D>,
}

#[pymethods]
impl NativeClusterTracker2D {
    #[new]
    fn new(config: NativeClusterTrackerConfig) -> PyResult<Self> {
        let tracker =
            ClusterTracker2D::new(native_tracker_config(config)?).map_err(tracking_error)?;
        Ok(Self {
            tracker: Mutex::new(tracker),
        })
    }

    fn step<'py>(
        &self,
        py: Python<'py>,
        centers: PyReadonlyArray2<'py, f32>,
        extents: PyReadonlyArray2<'py, f32>,
        mean_velocities: PyReadonlyArray1<'py, f32>,
        point_counts: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<NativeTrackerStepResult<'py>> {
        let (centers, center_shape) = tracking_matrix_input(centers, "Cluster centers")?;
        let (extents, extent_shape) = tracking_matrix_input(extents, "Cluster extents")?;
        if center_shape[1] != 3 {
            return Err(PyValueError::new_err(format!(
                "Cluster centers must have shape (N, 3); got ({}, {}).",
                center_shape[0], center_shape[1]
            )));
        }
        if extent_shape != center_shape {
            return Err(PyValueError::new_err(format!(
                "Cluster extents must have shape ({}, 3); got ({}, {}).",
                center_shape[0], extent_shape[0], extent_shape[1]
            )));
        }
        let mean_velocities = tracking_f32_vector_input(mean_velocities, "Cluster velocities")?;
        let point_counts = tracking_i64_vector_input(point_counts, "Cluster point counts")?;
        let tracker = &self.tracker;
        let result = py
            .detach(move || {
                let mut tracker = tracker
                    .lock()
                    .map_err(|_| TrackingError::StateLockPoisoned)?;
                tracker.step(ClusterMeasurements {
                    centers: &centers,
                    extents: &extents,
                    mean_velocities: &mean_velocities,
                    point_counts: &point_counts,
                })
            })
            .map_err(tracking_error)?;
        tracker_step_result_array(py, result, 2)
    }
}

#[pyclass]
struct NativePointTracker2D {
    tracker: Mutex<PointTracker2D>,
}

#[pymethods]
impl NativePointTracker2D {
    #[new]
    fn new(config: NativeMeasurementTrackerConfig) -> PyResult<Self> {
        let (tracker_config, allocation_clustering) = config;
        let tracker = PointTracker2D::new(
            native_tracker_config(tracker_config)?,
            dbscan_config(allocation_clustering)?,
        );
        Ok(Self {
            tracker: Mutex::new(tracker),
        })
    }

    fn step<'py>(
        &self,
        py: Python<'py>,
        coordinates: PyReadonlyArray2<'py, f32>,
        velocities: PyReadonlyArray1<'py, f32>,
        snrs: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<NativeTrackerStepResult<'py>> {
        let (coordinates, velocities, snrs) =
            point_measurement_input(coordinates, velocities, snrs)?;
        let tracker = &self.tracker;
        let result = py
            .detach(move || {
                let mut tracker = tracker
                    .lock()
                    .map_err(|_| TrackingError::StateLockPoisoned)?;
                tracker.step(PointMeasurements {
                    coordinates: &coordinates,
                    velocities: &velocities,
                    snrs: &snrs,
                })
            })
            .map_err(tracking_error)?;
        tracker_step_result_array(py, result, 2)
    }
}

#[pyclass]
struct NativePointTracker3D {
    tracker: Mutex<GTrack3D>,
}

#[pymethods]
impl NativePointTracker3D {
    #[new]
    fn new(config: NativeMeasurementTracker3DConfig) -> PyResult<Self> {
        let (tracker_config, allocation_clustering) = config;
        let tracker = GTrack3D::new(
            native_tracker_3d_config(tracker_config)?,
            dbscan_config(allocation_clustering)?,
        );
        Ok(Self {
            tracker: Mutex::new(tracker),
        })
    }

    fn step<'py>(
        &self,
        py: Python<'py>,
        coordinates: PyReadonlyArray2<'py, f32>,
        velocities: PyReadonlyArray1<'py, f32>,
        snrs: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<NativeGTrack3DStepResult<'py>> {
        let (coordinates, velocities, snrs) =
            point_measurement_input(coordinates, velocities, snrs)?;
        let tracker = &self.tracker;
        let (result, diagnostics) = py
            .detach(move || {
                let mut tracker = tracker
                    .lock()
                    .map_err(|_| TrackingError::StateLockPoisoned)?;
                let result = tracker.step(PointMeasurements {
                    coordinates: &coordinates,
                    velocities: &velocities,
                    snrs: &snrs,
                })?;
                Ok::<_, TrackingError>((result, tracker.diagnostics()))
            })
            .map_err(tracking_error)?;
        Ok((
            tracker_step_result_array(py, result, 3)?,
            gtrack3d_diagnostics(diagnostics),
        ))
    }
}

#[pyfunction]
fn summarize_tracking_metrics<'py>(
    py: Python<'py>,
    arrays: NativeTrackingMetricsInput<'py>,
    scenery_boxes: Option<Vec<NativeTrackingBox>>,
    frame_index_offset: usize,
) -> PyResult<NativeTrackingMetricsResult<'py>> {
    let (frame_offsets, track_ids, positions, velocities, status_codes) = arrays;
    let frame_offsets =
        tracking_usize_vector_input(frame_offsets, "Tracking metric frame offsets")?;
    let track_ids = tracking_i64_vector_input(track_ids, "Tracking metric track IDs")?;
    let (positions, position_shape) =
        tracking_matrix_input(positions, "Tracking metric positions")?;
    let (velocities, velocity_shape) =
        tracking_matrix_input(velocities, "Tracking metric velocities")?;
    if position_shape[1] != 3 || velocity_shape[1] != 3 {
        return Err(PyValueError::new_err(
            "Tracking metric positions and velocities must have shape (N, 3).",
        ));
    }
    if position_shape != velocity_shape {
        return Err(PyValueError::new_err(
            "Tracking metric positions and velocities must have the same shape.",
        ));
    }
    let status_codes = tracking_u8_vector_input(status_codes, "Tracking metric status codes")?;
    let scenery_boxes = scenery_boxes
        .map(|boxes| {
            boxes
                .into_iter()
                .map(|(x_min_m, x_max_m, y_min_m, y_max_m)| {
                    Box2D::new(x_min_m, x_max_m, y_min_m, y_max_m)
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()
        .map_err(tracking_error)?;
    let metrics = py
        .detach(move || {
            native_summarize_tracking_metrics(TrackingMetricsInput {
                frame_offsets: &frame_offsets,
                track_ids: &track_ids,
                positions: &positions,
                velocities: &velocities,
                status_codes: &status_codes,
                scenery_boxes: scenery_boxes.as_deref(),
                frame_index_offset,
            })
        })
        .map_err(tracking_metrics_error)?;
    tracking_metrics_result_array(py, metrics)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeClusterTracker2D>()?;
    module.add_class::<NativePointTracker2D>()?;
    module.add_class::<NativePointTracker3D>()?;
    module.add_function(wrap_pyfunction!(summarize_tracking_metrics, module)?)?;
    Ok(())
}

fn tracking_matrix_input(
    values: PyReadonlyArray2<'_, f32>,
    name: &str,
) -> PyResult<(Vec<f32>, [usize; 2])> {
    if !values.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{name} must be a C-contiguous float32 matrix."
        )));
    }
    let shape = values.shape();
    let values = values
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a contiguous float32 matrix.")))?
        .to_vec();
    Ok((values, [shape[0], shape[1]]))
}

fn point_measurement_input(
    coordinates: PyReadonlyArray2<'_, f32>,
    velocities: PyReadonlyArray1<'_, f32>,
    snrs: PyReadonlyArray1<'_, f32>,
) -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let (coordinates, coordinate_shape) = tracking_matrix_input(coordinates, "Point coordinates")?;
    if coordinate_shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "Point coordinates must have shape (N, 3); got ({}, {}).",
            coordinate_shape[0], coordinate_shape[1]
        )));
    }
    let velocities = tracking_f32_vector_input(velocities, "Point velocities")?;
    let snrs = tracking_f32_vector_input(snrs, "Point SNR")?;
    for (name, length) in [("velocities", velocities.len()), ("SNR", snrs.len())] {
        if length != coordinate_shape[0] {
            return Err(PyValueError::new_err(format!(
                "Point {name} must contain {} values; got {length}.",
                coordinate_shape[0]
            )));
        }
    }
    Ok((coordinates, velocities, snrs))
}

fn tracking_f32_vector_input(values: PyReadonlyArray1<'_, f32>, name: &str) -> PyResult<Vec<f32>> {
    if !values.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{name} must be a C-contiguous float32 vector."
        )));
    }
    values
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a contiguous float32 vector.")))
        .map(ToOwned::to_owned)
}

fn tracking_i64_vector_input(values: PyReadonlyArray1<'_, i64>, name: &str) -> PyResult<Vec<i64>> {
    if !values.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{name} must be a C-contiguous int64 vector."
        )));
    }
    values
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a contiguous int64 vector.")))
        .map(ToOwned::to_owned)
}

fn tracking_usize_vector_input(
    values: PyReadonlyArray1<'_, i64>,
    name: &str,
) -> PyResult<Vec<usize>> {
    let values = tracking_i64_vector_input(values, name)?;
    values
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            usize::try_from(value).map_err(|_| {
                PyValueError::new_err(format!("{name} value {index} must be non-negative."))
            })
        })
        .collect()
}

fn tracking_u8_vector_input(values: PyReadonlyArray1<'_, u8>, name: &str) -> PyResult<Vec<u8>> {
    if !values.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{name} must be a C-contiguous uint8 vector."
        )));
    }
    values
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a contiguous uint8 vector.")))
        .map(ToOwned::to_owned)
}

fn tracking_error(error: TrackingError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn tracking_metrics_error(error: TrackingMetricsError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn native_tracker_config(config: NativeClusterTrackerConfig) -> PyResult<Tracker2DConfig> {
    let (dynamics, gating, allocation, lifecycle, scenery, max_tracks) = config;
    let (
        frame_period_s,
        max_acceleration_mps2,
        measurement_noise_m,
        initial_velocity_std_mps,
        extent_covariance_smoothing,
        angle_noise_rad,
        doppler_noise_mps,
        max_velocity_mps,
    ) = dynamics;
    let (boundary_boxes, static_boxes, outside_max_frames) = scenery;
    let boundary_boxes = boundary_boxes
        .into_iter()
        .map(|(x_min_m, x_max_m, y_min_m, y_max_m)| Box2D::new(x_min_m, x_max_m, y_min_m, y_max_m))
        .collect::<Result<Vec<_>, _>>()
        .map_err(tracking_error)?;
    let static_boxes = static_boxes
        .into_iter()
        .map(|(x_min_m, x_max_m, y_min_m, y_max_m)| Box2D::new(x_min_m, x_max_m, y_min_m, y_max_m))
        .collect::<Result<Vec<_>, _>>()
        .map_err(tracking_error)?;
    let dynamics = TrackerDynamicsConfig::new(
        frame_period_s,
        [max_acceleration_mps2.0, max_acceleration_mps2.1],
        measurement_noise_m,
        initial_velocity_std_mps,
        extent_covariance_smoothing,
    )
    .and_then(|dynamics| {
        dynamics.with_polar_measurement(angle_noise_rad, doppler_noise_mps, max_velocity_mps)
    })
    .map_err(tracking_error)?;
    let gating = native_gating_config(gating)?;
    let allocation = native_allocation_config(allocation)?;
    let lifecycle = native_lifecycle_config(lifecycle)?;
    let scenery = TrackSceneryConfig::new(boundary_boxes, outside_max_frames)
        .map_err(tracking_error)?
        .with_static_boxes(static_boxes);
    Tracker2DConfig::new(dynamics, gating, allocation, lifecycle, scenery, max_tracks)
        .map_err(tracking_error)
}

fn native_tracker_3d_config(config: NativeTracker3DConfig) -> PyResult<Tracker3DConfig> {
    let (dynamics, gating, allocation, lifecycle, scenery, max_tracks) = config;
    let (
        frame_period_s,
        max_acceleration_mps2,
        measurement_noise_m,
        initial_velocity_std_mps,
        extent_covariance_smoothing,
        angle_noise_rad,
        elevation_noise_rad,
        doppler_noise_mps,
        max_velocity_mps,
    ) = dynamics;
    let (boundary_boxes, static_boxes, outside_max_frames) = scenery;
    let boundary_boxes = boundary_boxes
        .into_iter()
        .map(|(x_min, x_max, y_min, y_max, z_min, z_max)| {
            Box3D::new(x_min, x_max, y_min, y_max, z_min, z_max)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(tracking_error)?;
    let static_boxes = static_boxes
        .into_iter()
        .map(|(x_min, x_max, y_min, y_max, z_min, z_max)| {
            Box3D::new(x_min, x_max, y_min, y_max, z_min, z_max)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(tracking_error)?;
    let dynamics = TrackerDynamics3DConfig::new(
        frame_period_s,
        [
            max_acceleration_mps2.0,
            max_acceleration_mps2.1,
            max_acceleration_mps2.2,
        ],
        measurement_noise_m,
        initial_velocity_std_mps,
        extent_covariance_smoothing,
    )
    .and_then(|dynamics| {
        dynamics.with_spherical_measurement(
            angle_noise_rad,
            elevation_noise_rad,
            doppler_noise_mps,
            max_velocity_mps,
        )
    })
    .map_err(tracking_error)?;
    let gating = native_gating_config(gating)?;
    let allocation = native_allocation_config(allocation)?;
    let lifecycle = native_lifecycle_config(lifecycle)?;
    let scenery = TrackScenery3DConfig::new(boundary_boxes, outside_max_frames)
        .map_err(tracking_error)?
        .with_static_boxes(static_boxes);
    Tracker3DConfig::new(dynamics, gating, allocation, lifecycle, scenery, max_tracks)
        .map_err(tracking_error)
}

fn native_gating_config(config: NativeTrackerGatingConfig) -> PyResult<TrackGatingConfig> {
    let (max_distance_m, max_radial_velocity_difference_mps, max_mahalanobis_distance) = config;
    TrackGatingConfig::new(
        max_distance_m,
        max_radial_velocity_difference_mps,
        max_mahalanobis_distance,
    )
    .map_err(tracking_error)
}

fn native_allocation_config(
    config: NativeTrackerAllocationConfig,
) -> PyResult<TrackAllocationConfig> {
    let (
        min_points,
        min_abs_radial_velocity_mps,
        min_total_snr,
        max_new_tracks_per_frame,
        min_separation_m,
    ) = config;
    TrackAllocationConfig::new(
        min_points,
        min_abs_radial_velocity_mps,
        min_total_snr,
        max_new_tracks_per_frame,
        min_separation_m,
    )
    .map_err(tracking_error)
}

fn native_lifecycle_config(config: NativeTrackerLifecycleConfig) -> PyResult<TrackLifecycleConfig> {
    let (
        confirmation_hits,
        tentative_max_misses,
        confirmed_max_misses,
        min_update_points,
        static_max_misses,
        exit_max_misses,
        static_speed_threshold_mps,
    ) = config;
    TrackLifecycleConfig::new(
        confirmation_hits,
        tentative_max_misses,
        confirmed_max_misses,
        min_update_points,
    )
    .and_then(|lifecycle| {
        lifecycle.with_scene_miss_limits(
            static_max_misses,
            exit_max_misses,
            static_speed_threshold_mps,
        )
    })
    .map_err(tracking_error)
}

fn tracker_step_result_array(
    py: Python<'_>,
    result: TrackStepResult,
    covariance_size: usize,
) -> PyResult<NativeTrackerStepResult<'_>> {
    let track_count = result.track_ids.len();
    let positions = Array2::from_shape_vec((track_count, 3), result.positions)
        .map_err(|_| PyValueError::new_err("Native tracker positions shape is invalid."))?
        .into_pyarray(py);
    let velocities = Array2::from_shape_vec((track_count, 3), result.velocities)
        .map_err(|_| PyValueError::new_err("Native tracker velocities shape is invalid."))?
        .into_pyarray(py);
    let position_covariances = Array3::from_shape_vec(
        (track_count, covariance_size, covariance_size),
        result.position_covariances,
    )
    .map_err(|_| PyValueError::new_err("Native tracker position covariance shape is invalid."))?
    .into_pyarray(py);
    let extent_covariances = Array3::from_shape_vec(
        (track_count, covariance_size, covariance_size),
        result.extent_covariances,
    )
    .map_err(|_| PyValueError::new_err("Native tracker extent covariance shape is invalid."))?
    .into_pyarray(py);
    Ok((
        Array1::from_vec(result.track_ids).into_pyarray(py),
        positions,
        velocities,
        position_covariances,
        extent_covariances,
        Array1::from_vec(result.statuses).into_pyarray(py),
        Array1::from_vec(result.ages).into_pyarray(py),
        Array1::from_vec(result.missed_counts).into_pyarray(py),
        Array1::from_vec(result.observation_track_ids).into_pyarray(py),
    ))
}

fn gtrack3d_diagnostics(value: GTrack3DDiagnostics) -> NativeGTrack3DDiagnostics {
    (
        (value.frames, value.points, value.outside_points),
        (
            value.distance_gate_misses,
            value.doppler_gate_misses,
            value.mahalanobis_gate_misses,
        ),
        (value.empty_updates, value.partial_updates),
        (value.allocations, value.confirmations, value.reactivations),
        (
            value.tentative_deletions,
            value.coasting_deletions,
            value.outside_deletions,
        ),
    )
}

fn tracking_metrics_result_array(
    py: Python<'_>,
    metrics: TrackingSequenceMetrics,
) -> PyResult<NativeTrackingMetricsResult<'_>> {
    let track_count = metrics.tracks.len();
    let mut track_ids = Vec::with_capacity(track_count);
    let mut observed_frames = Vec::with_capacity(track_count);
    let mut confirmed_frames = Vec::with_capacity(track_count);
    let mut first_frame_indices = Vec::with_capacity(track_count);
    let mut last_frame_indices = Vec::with_capacity(track_count);
    let mut first_positions = Vec::with_capacity(track_count * 3);
    let mut last_positions = Vec::with_capacity(track_count * 3);
    let mut median_positions = Vec::with_capacity(track_count * 3);
    let mut displacement = Vec::with_capacity(track_count);
    let mut path_length = Vec::with_capacity(track_count);
    let mut median_speed = Vec::with_capacity(track_count);
    let mut max_speed = Vec::with_capacity(track_count);
    let mut interval_offsets = Vec::with_capacity(track_count + 1);
    let mut intervals = Vec::new();
    let mut in_scenery_frames = Vec::with_capacity(track_count);
    let mut outside_scenery_frames = Vec::with_capacity(track_count);
    interval_offsets.push(0_i64);
    for track in metrics.tracks {
        track_ids.push(track.track_id);
        observed_frames.push(usize_to_i64(track.observed_frames, "observed frame count")?);
        confirmed_frames.push(usize_to_i64(
            track.confirmed_frames,
            "confirmed frame count",
        )?);
        first_frame_indices.push(usize_to_i64(track.first_frame_index, "first frame index")?);
        last_frame_indices.push(usize_to_i64(track.last_frame_index, "last frame index")?);
        first_positions.extend(track.first_position_m);
        last_positions.extend(track.last_position_m);
        median_positions.extend(track.median_position_m);
        displacement.push(track.displacement_m);
        path_length.push(track.path_length_m);
        median_speed.push(track.median_speed_mps);
        max_speed.push(track.max_speed_mps);
        for [start, stop] in track.confirmed_intervals {
            intervals.push(usize_to_i64(start, "confirmed interval start")?);
            intervals.push(usize_to_i64(stop, "confirmed interval stop")?);
        }
        interval_offsets.push(usize_to_i64(
            intervals.len() / 2,
            "confirmed interval count",
        )?);
        in_scenery_frames.push(optional_usize_to_i64(
            track.in_scenery_frames,
            "in-scenery frame count",
        )?);
        outside_scenery_frames.push(optional_usize_to_i64(
            track.outside_scenery_frames,
            "outside-scenery frame count",
        )?);
    }
    let interval_count = intervals.len() / 2;
    Ok((
        (
            metrics.num_frames,
            metrics.frames_with_tracks,
            metrics.frames_with_confirmed_tracks,
            metrics.max_concurrent_tracks,
        ),
        (
            Array1::from_vec(track_ids).into_pyarray(py),
            Array1::from_vec(observed_frames).into_pyarray(py),
            Array1::from_vec(confirmed_frames).into_pyarray(py),
            Array1::from_vec(first_frame_indices).into_pyarray(py),
            Array1::from_vec(last_frame_indices).into_pyarray(py),
            Array2::from_shape_vec((track_count, 3), first_positions)
                .map_err(|_| {
                    PyValueError::new_err("Native tracking first-position shape is invalid.")
                })?
                .into_pyarray(py),
            Array2::from_shape_vec((track_count, 3), last_positions)
                .map_err(|_| {
                    PyValueError::new_err("Native tracking last-position shape is invalid.")
                })?
                .into_pyarray(py),
            Array2::from_shape_vec((track_count, 3), median_positions)
                .map_err(|_| {
                    PyValueError::new_err("Native tracking median-position shape is invalid.")
                })?
                .into_pyarray(py),
        ),
        (
            Array1::from_vec(displacement).into_pyarray(py),
            Array1::from_vec(path_length).into_pyarray(py),
            Array1::from_vec(median_speed).into_pyarray(py),
            Array1::from_vec(max_speed).into_pyarray(py),
        ),
        (
            Array1::from_vec(interval_offsets).into_pyarray(py),
            Array2::from_shape_vec((interval_count, 2), intervals)
                .map_err(|_| PyValueError::new_err("Native tracking interval shape is invalid."))?
                .into_pyarray(py),
            Array1::from_vec(in_scenery_frames).into_pyarray(py),
            Array1::from_vec(outside_scenery_frames).into_pyarray(py),
        ),
    ))
}

fn usize_to_i64(value: usize, name: &str) -> PyResult<i64> {
    i64::try_from(value)
        .map_err(|_| PyValueError::new_err(format!("Native tracking {name} exceeds int64.")))
}

fn optional_usize_to_i64(value: Option<usize>, name: &str) -> PyResult<i64> {
    value.map_or(Ok(-1), |value| usize_to_i64(value, name))
}

type NativeTrackerDynamicsConfig = (f64, (f64, f64), f64, f64, f64, f64, f64, f64);
type NativeTrackerGatingConfig = (f64, Option<f64>, Option<f64>);
type NativeTrackerAllocationConfig = (usize, f64, Option<f64>, Option<usize>, Option<f64>);
type NativeTrackerLifecycleConfig = (
    usize,
    usize,
    usize,
    usize,
    Option<usize>,
    Option<usize>,
    f64,
);
type NativeTrackingBox = (f64, f64, f64, f64);
type NativeTrackerSceneryConfig = (Vec<NativeTrackingBox>, Vec<NativeTrackingBox>, usize);
type NativeClusterTrackerConfig = (
    NativeTrackerDynamicsConfig,
    NativeTrackerGatingConfig,
    NativeTrackerAllocationConfig,
    NativeTrackerLifecycleConfig,
    NativeTrackerSceneryConfig,
    usize,
);
type NativeMeasurementTrackerConfig = (NativeClusterTrackerConfig, NativeDbscanConfig);
type NativeTrackerDynamics3DConfig = (f64, (f64, f64, f64), f64, f64, f64, f64, f64, f64, f64);
type NativeTrackingBox3D = (f64, f64, f64, f64, f64, f64);
type NativeTrackerScenery3DConfig = (Vec<NativeTrackingBox3D>, Vec<NativeTrackingBox3D>, usize);
type NativeTracker3DConfig = (
    NativeTrackerDynamics3DConfig,
    NativeTrackerGatingConfig,
    NativeTrackerAllocationConfig,
    NativeTrackerLifecycleConfig,
    NativeTrackerScenery3DConfig,
    usize,
);
type NativeMeasurementTracker3DConfig = (NativeTracker3DConfig, NativeDbscanConfig);
type NativeTrackingMetricsInput<'py> = (
    PyReadonlyArray1<'py, i64>,
    PyReadonlyArray1<'py, i64>,
    PyReadonlyArray2<'py, f32>,
    PyReadonlyArray2<'py, f32>,
    PyReadonlyArray1<'py, u8>,
);
type NativeTrackerStepResult<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray3<f32>>,
    Bound<'py, PyArray3<f32>>,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
);
type NativeGTrack3DDiagnostics = (
    (u64, u64, u64),
    (u64, u64, u64),
    (u64, u64),
    (u64, u64, u64),
    (u64, u64, u64),
);
type NativeGTrack3DStepResult<'py> = (NativeTrackerStepResult<'py>, NativeGTrack3DDiagnostics);
type NativeTrackingMetricsHeader = (usize, usize, usize, usize);
type NativeTrackingMetricsIdentity<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
);
type NativeTrackingMetricsMotion<'py> = (
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
);
type NativeTrackingMetricsIntervals<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
);
type NativeTrackingMetricsResult<'py> = (
    NativeTrackingMetricsHeader,
    NativeTrackingMetricsIdentity<'py>,
    NativeTrackingMetricsMotion<'py>,
    NativeTrackingMetricsIntervals<'py>,
);
