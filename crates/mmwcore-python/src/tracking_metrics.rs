//! Shared tracking comparison metrics; no prototype tracker bindings.
use mmwcore::{
    Box2D, TrackingMetricsError, TrackingMetricsInput, TrackingSequenceMetrics,
    summarize_tracking_metrics as native_summarize_tracking_metrics,
};
use numpy::ndarray::{Array1, Array2};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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
        .map_err(tracking_metrics_error)?;
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

fn tracking_metrics_error(error: TrackingMetricsError) -> PyErr {
    PyValueError::new_err(error.to_string())
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

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(summarize_tracking_metrics, module)?)?;
    Ok(())
}

type NativeTrackingBox = (f64, f64, f64, f64);

type NativeTrackingMetricsInput<'py> = (
    PyReadonlyArray1<'py, i64>,
    PyReadonlyArray1<'py, i64>,
    PyReadonlyArray2<'py, f32>,
    PyReadonlyArray2<'py, f32>,
    PyReadonlyArray1<'py, u8>,
);

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
