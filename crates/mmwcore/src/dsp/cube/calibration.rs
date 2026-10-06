use num_complex::Complex32;

use super::{
    CubeTransformError, contiguous_strides, coordinate, validate_axis, validate_distinct_axes,
    validate_shape,
};

pub fn apply_time_domain_channel_calibration_complex(
    data: &[Complex32],
    shape: &[usize],
    tx_axis: usize,
    rx_axis: usize,
    sample_axis: usize,
    frequencies_rad_per_sample: &[f32],
    corrections: &[Complex32],
) -> Result<Vec<Complex32>, CubeTransformError> {
    validate_shape(data, shape)?;
    validate_distinct_axes(shape, &[tx_axis, rx_axis, sample_axis])?;

    let num_tx = shape[tx_axis];
    let num_rx = shape[rx_axis];
    let expected = num_tx
        .checked_mul(num_rx)
        .ok_or(CubeTransformError::ShapeOverflow)?;
    if frequencies_rad_per_sample.len() != expected {
        return Err(CubeTransformError::CalibrationShapeMismatch {
            expected,
            actual: frequencies_rad_per_sample.len(),
        });
    }
    if corrections.len() != expected {
        return Err(CubeTransformError::CalibrationShapeMismatch {
            expected,
            actual: corrections.len(),
        });
    }

    let strides = contiguous_strides(shape)?;
    let mut output = Vec::with_capacity(data.len());
    for (flat_index, sample) in data.iter().copied().enumerate() {
        let tx = coordinate(flat_index, strides[tx_axis], num_tx);
        let rx = coordinate(flat_index, strides[rx_axis], num_rx);
        let sample_index = coordinate(flat_index, strides[sample_axis], shape[sample_axis]);
        let calibration_index = tx * num_rx + rx;
        let phase = Complex32::from_polar(
            1.0,
            frequencies_rad_per_sample[calibration_index] * sample_index as f32,
        );
        output.push(sample * corrections[calibration_index] * phase);
    }

    Ok(output)
}

pub fn apply_virtual_channel_calibration_complex(
    data: &[Complex32],
    shape: &[usize],
    virtual_axis: usize,
    coefficients: &[Complex32],
) -> Result<Vec<Complex32>, CubeTransformError> {
    validate_shape(data, shape)?;
    validate_axis(shape, virtual_axis)?;
    let axis_length = shape[virtual_axis];
    if coefficients.len() != axis_length {
        return Err(CubeTransformError::CalibrationShapeMismatch {
            expected: axis_length,
            actual: coefficients.len(),
        });
    }

    let stride = contiguous_strides(shape)?[virtual_axis];
    Ok(data
        .iter()
        .copied()
        .enumerate()
        .map(|(flat_index, value)| {
            value * coefficients[coordinate(flat_index, stride, axis_length)]
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::{
        apply_time_domain_channel_calibration_complex, apply_virtual_channel_calibration_complex,
    };
    use num_complex::Complex32;
    use std::f32::consts::PI;

    #[test]
    fn applies_channel_calibration_without_axis_moves() {
        let data = vec![Complex32::new(1.0, 0.0); 12];
        let output = apply_time_domain_channel_calibration_complex(
            &data,
            &[1, 1, 2, 2, 3],
            2,
            3,
            4,
            &[0.0, PI / 2.0, PI, 0.0],
            &[
                Complex32::new(1.0, 0.0),
                Complex32::new(0.0, 1.0),
                Complex32::new(0.5, 0.0),
                Complex32::new(0.0, -1.0),
            ],
        )
        .unwrap();

        assert_complex_close(output[0], Complex32::new(1.0, 0.0));
        assert_complex_close(output[4], Complex32::new(-1.0, 0.0));
        assert_complex_close(output[8], Complex32::new(0.5, 0.0));

        let calibrated = apply_virtual_channel_calibration_complex(
            &[Complex32::new(2.0, 0.0), Complex32::new(0.0, 0.5)],
            &[1, 1, 2, 1],
            2,
            &[Complex32::new(0.5, 0.0), Complex32::new(0.0, -2.0)],
        )
        .unwrap();
        assert_complex_slice_close(
            &calibrated,
            &[Complex32::new(1.0, 0.0), Complex32::new(1.0, 0.0)],
        );
    }

    fn assert_complex_slice_close(actual: &[Complex32], expected: &[Complex32]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert_complex_close(actual, expected);
        }
    }

    fn assert_complex_close(actual: Complex32, expected: Complex32) {
        let delta = actual - expected;
        assert!(delta.re.abs() < 1e-5 && delta.im.abs() < 1e-5);
    }
}
