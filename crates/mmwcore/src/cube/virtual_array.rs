use std::collections::HashSet;
use std::f32::consts::PI;

use num_complex::Complex32;

use super::{
    CubeTransformError, checked_product, contiguous_strides, coordinate, validate_axis,
    validate_distinct_axes, validate_shape,
};

pub fn map_tdm_virtual_array_complex(
    data: &[Complex32],
    shape: &[usize],
    chirp_axis: usize,
    rx_axis: usize,
    num_tx: usize,
) -> Result<(Vec<Complex32>, Vec<usize>), CubeTransformError> {
    validate_shape(data, shape)?;
    validate_distinct_axes(shape, &[chirp_axis, rx_axis])?;
    if num_tx == 0 {
        return Err(CubeTransformError::InvalidTdmTxCount { num_tx });
    }

    let num_chirps = shape[chirp_axis];
    if num_chirps % num_tx != 0 {
        return Err(CubeTransformError::IncompleteTdmLoops {
            chirps: num_chirps,
            num_tx,
        });
    }
    let num_loops = num_chirps / num_tx;
    let num_rx = shape[rx_axis];
    let num_virtual = num_rx
        .checked_mul(num_tx)
        .ok_or(CubeTransformError::ShapeOverflow)?;
    let mut output_shape = shape.to_vec();
    output_shape[chirp_axis] = num_loops;
    output_shape[rx_axis] = num_virtual;
    let output_length = checked_product(&output_shape)?;
    if output_length != data.len() {
        return Err(CubeTransformError::ShapeSizeMismatch {
            expected: output_length,
            actual: data.len(),
        });
    }

    let input_strides = contiguous_strides(shape)?;
    let output_strides = contiguous_strides(&output_shape)?;
    let mut output = vec![Complex32::new(0.0, 0.0); data.len()];
    for (output_index, output_value) in output.iter_mut().enumerate() {
        let loop_index = coordinate(
            output_index,
            output_strides[chirp_axis],
            output_shape[chirp_axis],
        );
        let virtual_index =
            coordinate(output_index, output_strides[rx_axis], output_shape[rx_axis]);
        let tx = virtual_index / num_rx;
        let rx = virtual_index % num_rx;
        let mut input_index = 0;
        for axis in 0..shape.len() {
            let output_coordinate =
                coordinate(output_index, output_strides[axis], output_shape[axis]);
            let input_coordinate = if axis == chirp_axis {
                loop_index * num_tx + tx
            } else if axis == rx_axis {
                rx
            } else {
                output_coordinate
            };
            input_index += input_coordinate * input_strides[axis];
        }
        *output_value = data[input_index];
    }

    Ok((output, output_shape))
}

pub fn compensate_tdm_doppler_phase_complex(
    data: &[Complex32],
    shape: &[usize],
    doppler_axis: usize,
    virtual_axis: usize,
    num_tx: usize,
    num_rx: usize,
    fftshift: bool,
) -> Result<Vec<Complex32>, CubeTransformError> {
    validate_shape(data, shape)?;
    validate_distinct_axes(shape, &[doppler_axis, virtual_axis])?;
    if num_tx == 0 {
        return Err(CubeTransformError::InvalidTdmTxCount { num_tx });
    }
    let expected_virtual = num_tx
        .checked_mul(num_rx)
        .ok_or(CubeTransformError::ShapeOverflow)?;
    let actual_virtual = shape[virtual_axis];
    if actual_virtual != expected_virtual {
        return Err(CubeTransformError::VirtualChannelMismatch {
            expected: expected_virtual,
            actual: actual_virtual,
        });
    }

    let num_doppler_bins = shape[doppler_axis];
    let strides = contiguous_strides(shape)?;
    let denominator = num_doppler_bins
        .checked_mul(num_tx)
        .ok_or(CubeTransformError::ShapeOverflow)? as f32;
    Ok(data
        .iter()
        .copied()
        .enumerate()
        .map(|(flat_index, value)| {
            let doppler_index = coordinate(flat_index, strides[doppler_axis], num_doppler_bins);
            let virtual_index = coordinate(flat_index, strides[virtual_axis], actual_virtual);
            let signed_bin = signed_doppler_bin(doppler_index, num_doppler_bins, fftshift);
            let tx_slot = virtual_index / num_rx;
            let phase = -2.0 * PI * signed_bin as f32 * tx_slot as f32 / denominator;
            value * Complex32::from_polar(1.0, phase)
        })
        .collect())
}

pub fn map_planar_aperture_complex(
    data: &[Complex32],
    shape: &[usize],
    virtual_axis: usize,
    grid_indices: &[(usize, usize)],
) -> Result<(Vec<Complex32>, Vec<usize>), CubeTransformError> {
    validate_shape(data, shape)?;
    validate_axis(shape, virtual_axis)?;
    let num_virtual = shape[virtual_axis];
    if grid_indices.len() != num_virtual {
        return Err(CubeTransformError::PlanarPositionMismatch {
            expected: num_virtual,
            actual: grid_indices.len(),
        });
    }

    let (azimuth_length, elevation_length) = aperture_shape(grid_indices)?;
    let mut output_shape = Vec::with_capacity(shape.len() + 1);
    for (axis, axis_length) in shape.iter().copied().enumerate() {
        if axis == virtual_axis {
            output_shape.push(azimuth_length);
            output_shape.push(elevation_length);
        } else {
            output_shape.push(axis_length);
        }
    }
    let output_length = checked_product(&output_shape)?;
    let input_strides = contiguous_strides(shape)?;
    let output_strides = contiguous_strides(&output_shape)?;
    let mut output = vec![Complex32::new(0.0, 0.0); output_length];
    let first_channels = first_planar_channels(grid_indices);

    for (input_index, value) in data.iter().copied().enumerate() {
        let channel = coordinate(input_index, input_strides[virtual_axis], num_virtual);
        if !first_channels[channel] {
            continue;
        }
        let (azimuth, elevation) = grid_indices[channel];
        let mut output_index = 0;
        let mut output_axis = 0;
        for axis in 0..shape.len() {
            if axis == virtual_axis {
                output_index += azimuth * output_strides[output_axis];
                output_axis += 1;
                output_index += elevation * output_strides[output_axis];
                output_axis += 1;
            } else {
                let input_coordinate = coordinate(input_index, input_strides[axis], shape[axis]);
                output_index += input_coordinate * output_strides[output_axis];
                output_axis += 1;
            }
        }
        output[output_index] = value;
    }

    Ok((output, output_shape))
}

pub fn select_virtual_subarray_complex(
    data: &[Complex32],
    shape: &[usize],
    virtual_axis: usize,
    indices: &[usize],
) -> Result<(Vec<Complex32>, Vec<usize>), CubeTransformError> {
    validate_shape(data, shape)?;
    validate_axis(shape, virtual_axis)?;
    if indices.is_empty() {
        return Err(CubeTransformError::SelectionMustNotBeEmpty);
    }
    let source_length = shape[virtual_axis];
    if let Some(&index) = indices.iter().find(|&&index| index >= source_length) {
        return Err(CubeTransformError::SelectionIndexOutOfBounds {
            index,
            axis_length: source_length,
        });
    }

    let mut output_shape = shape.to_vec();
    output_shape[virtual_axis] = indices.len();
    let input_strides = contiguous_strides(shape)?;
    let output_strides = contiguous_strides(&output_shape)?;
    let output_length = checked_product(&output_shape)?;
    let mut output = vec![Complex32::new(0.0, 0.0); output_length];
    for (output_index, output_value) in output.iter_mut().enumerate() {
        let selection_index = coordinate(
            output_index,
            output_strides[virtual_axis],
            output_shape[virtual_axis],
        );
        let mut input_index = 0;
        for axis in 0..shape.len() {
            let output_coordinate =
                coordinate(output_index, output_strides[axis], output_shape[axis]);
            let input_coordinate = if axis == virtual_axis {
                indices[selection_index]
            } else {
                output_coordinate
            };
            input_index += input_coordinate * input_strides[axis];
        }
        *output_value = data[input_index];
    }

    Ok((output, output_shape))
}

fn signed_doppler_bin(index: usize, length: usize, fftshift: bool) -> i64 {
    let unshifted_index = if fftshift {
        (index + length.div_ceil(2)) % length
    } else {
        index
    };
    if unshifted_index < length.div_ceil(2) {
        unshifted_index as i64
    } else {
        unshifted_index as i64 - length as i64
    }
}

fn aperture_shape(grid_indices: &[(usize, usize)]) -> Result<(usize, usize), CubeTransformError> {
    let max_azimuth = grid_indices
        .iter()
        .map(|&(azimuth, _)| azimuth)
        .max()
        .ok_or(CubeTransformError::PlanarPositionMismatch {
            expected: 1,
            actual: 0,
        })?;
    let max_elevation = grid_indices
        .iter()
        .map(|&(_, elevation)| elevation)
        .max()
        .ok_or(CubeTransformError::PlanarPositionMismatch {
            expected: 1,
            actual: 0,
        })?;
    let azimuth_length = max_azimuth
        .checked_add(1)
        .ok_or(CubeTransformError::ShapeOverflow)?;
    let elevation_length = max_elevation
        .checked_add(1)
        .ok_or(CubeTransformError::ShapeOverflow)?;
    Ok((azimuth_length, elevation_length))
}

fn first_planar_channels(grid_indices: &[(usize, usize)]) -> Vec<bool> {
    let mut seen = HashSet::new();
    grid_indices
        .iter()
        .map(|&position| seen.insert(position))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        compensate_tdm_doppler_phase_complex, map_planar_aperture_complex,
        map_tdm_virtual_array_complex, select_virtual_subarray_complex,
    };
    use num_complex::Complex32;
    use std::f32::consts::PI;

    #[test]
    fn maps_tdm_and_compensates_doppler_phase() {
        let data = (0..8)
            .map(|value| Complex32::new(value as f32, 0.0))
            .collect::<Vec<_>>();
        let (mapped, shape) = map_tdm_virtual_array_complex(&data, &[1, 4, 2, 1], 1, 2, 2).unwrap();
        assert_eq!(shape, [1, 2, 4, 1]);
        assert_complex_slice_close(
            &mapped,
            &(0..8)
                .map(|value| Complex32::new(value as f32, 0.0))
                .collect::<Vec<_>>(),
        );

        let signed_bins = [0.0, 1.0, -2.0, -1.0];
        let mut phase_shifted = vec![Complex32::new(1.0, 0.0); 8];
        for (doppler, &signed_bin) in signed_bins.iter().enumerate() {
            phase_shifted[doppler * 2 + 1] =
                Complex32::from_polar(1.0, 2.0 * PI * signed_bin / 8.0);
        }
        let compensated =
            compensate_tdm_doppler_phase_complex(&phase_shifted, &[1, 4, 2, 1], 1, 2, 2, 1, false)
                .unwrap();
        assert_complex_slice_close(&compensated, &[Complex32::new(1.0, 0.0); 8]);
    }

    #[test]
    fn scatters_planar_aperture_and_selects_virtual_channels() {
        let data = vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(99.0, 0.0),
            Complex32::new(4.0, 0.0),
        ];
        let (planar, planar_shape) =
            map_planar_aperture_complex(&data, &[1, 1, 4, 1], 2, &[(0, 0), (1, 0), (1, 0), (2, 1)])
                .unwrap();
        assert_eq!(planar_shape, [1, 1, 3, 2, 1]);
        assert_complex_slice_close(
            &planar,
            &[
                Complex32::new(1.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(2.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(4.0, 0.0),
            ],
        );

        let (selected, selected_shape) =
            select_virtual_subarray_complex(&data, &[1, 1, 4, 1], 2, &[3, 1]).unwrap();
        assert_eq!(selected_shape, [1, 1, 2, 1]);
        assert_complex_slice_close(
            &selected,
            &[Complex32::new(4.0, 0.0), Complex32::new(2.0, 0.0)],
        );
    }

    fn assert_complex_slice_close(actual: &[Complex32], expected: &[Complex32]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            let delta = actual - expected;
            assert!(delta.re.abs() < 1e-5 && delta.im.abs() < 1e-5);
        }
    }
}
