use num_complex::Complex32;

use super::{CubeTransformError, checked_product, validate_axis, validate_shape};

pub fn remove_static_clutter_complex(
    data: &[Complex32],
    shape: &[usize],
    axis: usize,
) -> Result<Vec<Complex32>, CubeTransformError> {
    validate_shape(data, shape)?;
    validate_axis(shape, axis)?;

    let outer = checked_product(&shape[..axis])?;
    let axis_length = shape[axis];
    let inner = checked_product(&shape[axis + 1..])?;
    let mut output = vec![Complex32::new(0.0, 0.0); data.len()];

    for outer_index in 0..outer {
        for inner_index in 0..inner {
            let base = outer_index * axis_length * inner + inner_index;
            let mut sum = Complex32::new(0.0, 0.0);
            for axis_index in 0..axis_length {
                sum += data[base + axis_index * inner];
            }
            let mean = sum / axis_length as f32;
            for axis_index in 0..axis_length {
                let index = base + axis_index * inner;
                output[index] = data[index] - mean;
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::remove_static_clutter_complex;
    use num_complex::Complex32;

    #[test]
    fn removes_static_clutter_on_named_axis_index() {
        let output = remove_static_clutter_complex(
            &[Complex32::new(1.0, 1.0), Complex32::new(3.0, 3.0)],
            &[1, 2, 1, 1],
            1,
        )
        .unwrap();

        assert_complex_slice_close(
            &output,
            &[Complex32::new(-1.0, -1.0), Complex32::new(1.0, 1.0)],
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
