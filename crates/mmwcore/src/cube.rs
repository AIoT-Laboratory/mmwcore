//! Named-axis complex-cube transforms on contiguous radar buffers.

use std::fmt;

mod calibration;
mod clutter;
mod virtual_array;

pub use calibration::{
    apply_time_domain_channel_calibration_complex, apply_virtual_channel_calibration_complex,
};
pub use clutter::remove_static_clutter_complex;
pub use virtual_array::{
    compensate_tdm_doppler_phase_complex, map_planar_aperture_complex,
    map_tdm_virtual_array_complex, select_virtual_subarray_complex,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CubeTransformError {
    EmptyShape,
    ZeroDimension { axis: usize },
    ShapeSizeMismatch { expected: usize, actual: usize },
    ShapeOverflow,
    AxisOutOfBounds { axis: usize, rank: usize },
    DuplicateAxes { first: usize, second: usize },
    InvalidTdmTxCount { num_tx: usize },
    IncompleteTdmLoops { chirps: usize, num_tx: usize },
    CalibrationShapeMismatch { expected: usize, actual: usize },
    VirtualChannelMismatch { expected: usize, actual: usize },
    SelectionMustNotBeEmpty,
    SelectionIndexOutOfBounds { index: usize, axis_length: usize },
    PlanarPositionMismatch { expected: usize, actual: usize },
}

impl fmt::Display for CubeTransformError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyShape => write!(formatter, "Radar cube shape must not be empty."),
            Self::ZeroDimension { axis } => {
                write!(
                    formatter,
                    "Radar cube axis {axis} must not have length zero."
                )
            }
            Self::ShapeSizeMismatch { expected, actual } => write!(
                formatter,
                "Radar cube buffer size {actual} does not match shape size {expected}."
            ),
            Self::ShapeOverflow => write!(formatter, "Radar cube shape overflows usize."),
            Self::AxisOutOfBounds { axis, rank } => {
                write!(formatter, "Radar cube axis {axis} is outside rank {rank}.")
            }
            Self::DuplicateAxes { first, second } => {
                write!(
                    formatter,
                    "Radar cube axes {first} and {second} must be distinct."
                )
            }
            Self::InvalidTdmTxCount { num_tx } => {
                write!(
                    formatter,
                    "TDM transmitter count must be positive; got {num_tx}."
                )
            }
            Self::IncompleteTdmLoops { chirps, num_tx } => write!(
                formatter,
                "Radar cube chirp count {chirps} does not contain complete TDM loops for {num_tx} transmitters."
            ),
            Self::CalibrationShapeMismatch { expected, actual } => write!(
                formatter,
                "Calibration coefficient count {actual} does not match expected {expected}."
            ),
            Self::VirtualChannelMismatch { expected, actual } => write!(
                formatter,
                "Virtual channel count {actual} does not match expected {expected}."
            ),
            Self::SelectionMustNotBeEmpty => {
                write!(formatter, "Virtual-channel selection must not be empty.")
            }
            Self::SelectionIndexOutOfBounds { index, axis_length } => write!(
                formatter,
                "Virtual-channel selection index {index} is outside axis length {axis_length}."
            ),
            Self::PlanarPositionMismatch { expected, actual } => write!(
                formatter,
                "Planar aperture position count {actual} does not match virtual-channel count {expected}."
            ),
        }
    }
}

impl std::error::Error for CubeTransformError {}

pub(crate) fn validate_shape<T>(data: &[T], shape: &[usize]) -> Result<(), CubeTransformError> {
    if shape.is_empty() {
        return Err(CubeTransformError::EmptyShape);
    }
    for (axis, &axis_length) in shape.iter().enumerate() {
        if axis_length == 0 {
            return Err(CubeTransformError::ZeroDimension { axis });
        }
    }
    let expected = checked_product(shape)?;
    if data.len() != expected {
        return Err(CubeTransformError::ShapeSizeMismatch {
            expected,
            actual: data.len(),
        });
    }
    Ok(())
}

pub(crate) fn validate_axis(shape: &[usize], axis: usize) -> Result<(), CubeTransformError> {
    if axis >= shape.len() {
        return Err(CubeTransformError::AxisOutOfBounds {
            axis,
            rank: shape.len(),
        });
    }
    Ok(())
}

pub(super) fn validate_distinct_axes(
    shape: &[usize],
    axes: &[usize],
) -> Result<(), CubeTransformError> {
    for &axis in axes {
        validate_axis(shape, axis)?;
    }
    for (index, &first) in axes.iter().enumerate() {
        if let Some(&second) = axes[index + 1..].iter().find(|&&second| second == first) {
            return Err(CubeTransformError::DuplicateAxes { first, second });
        }
    }
    Ok(())
}

pub(crate) fn checked_product(values: &[usize]) -> Result<usize, CubeTransformError> {
    values.iter().try_fold(1_usize, |product, &value| {
        product
            .checked_mul(value)
            .ok_or(CubeTransformError::ShapeOverflow)
    })
}

pub(super) fn contiguous_strides(shape: &[usize]) -> Result<Vec<usize>, CubeTransformError> {
    let mut strides = vec![1; shape.len()];
    let mut stride = 1_usize;
    for axis in (0..shape.len()).rev() {
        strides[axis] = stride;
        stride = stride
            .checked_mul(shape[axis])
            .ok_or(CubeTransformError::ShapeOverflow)?;
    }
    Ok(strides)
}

pub(super) fn coordinate(flat_index: usize, stride: usize, axis_length: usize) -> usize {
    flat_index / stride % axis_length
}
