// Copyright (C) 2024 Texas Instruments Incorporated.
// Rust adaptation of the pinned 3DA kernels; see ../TI-LICENSE.txt.
// Keep f32 evaluation order: association decisions depend on these values.
use super::Config;
use std::f32::consts::PI;

mod tables;

pub type Matrix<const R: usize, const C: usize> = [[f32; C]; R];

pub fn mul<const R: usize, const K: usize, const C: usize>(
    a: &Matrix<R, K>,
    b: &Matrix<K, C>,
) -> Matrix<R, C> {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let mut sum = 0.0;
            for k in 0..K {
                sum += a[i][k] * b[k][j];
            }
            sum
        })
    })
}

pub fn mul_transpose<const R: usize, const K: usize, const C: usize>(
    a: &Matrix<R, K>,
    b: &Matrix<C, K>,
) -> Matrix<R, C> {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let mut sum = 0.0;
            for k in 0..K {
                sum += a[i][k] * b[j][k];
            }
            sum
        })
    })
}

pub fn sincosd(theta: f32) -> (f32, f32) {
    // TI uses a one-degree lookup, including its asymmetric negative-angle
    // interpolation. Replacing this with sin_cos silently changes the tracker.
    // Bound extreme angles before indexing; the C table has no bounds check.
    let theta = if (-179.0..179.0).contains(&theta) {
        theta
    } else {
        (theta + 180.0).rem_euclid(360.0) - 180.0
    };
    if !(-179.0..179.0).contains(&theta) {
        return theta.to_radians().sin_cos();
    }
    let i = (theta + 179.0) as usize;
    let fraction = theta - theta.trunc() + if theta > 0.0 { 0.0 } else { 1.0 };
    let interpolate = |table: &[f32; 360]| table[i] + fraction * (table[i + 1] - table[i]);
    (interpolate(&tables::SIN), interpolate(&tables::COS))
}

pub fn position(u: &[f32; 4]) -> [f32; 3] {
    let (sa, ca) = sincosd(u[1] * (180.0 / PI));
    let (se, ce) = sincosd(u[2] * (180.0 / PI));
    [u[0] * ce * sa, u[0] * ce * ca, u[0] * se]
}

pub fn spherical(p: &[f32]) -> [f32; 4] {
    let [x, y, z] = [p[0], p[1], p[2]];
    [
        (x * x + y * y + z * z).sqrt(),
        atan(x / y),
        atan(z / (x * x + y * y).sqrt()),
        0.0,
    ]
}

fn atan(value: f32) -> f32 {
    // Keep TI's f32 input/output, but avoid host atanf rounding differences
    // accumulating in the covariance update across Linux and Windows.
    f64::from(value).atan() as f32
}

pub fn measurement(s: &[f32; 9]) -> [f32; 4] {
    let mut u = spherical(s);
    u[1] = if s[1] == 0.0 {
        PI / 2.0
    } else if s[1] > 0.0 {
        u[1]
    } else {
        u[1] + PI
    };
    u[3] = (s[0] * s[3] + s[1] * s[4] + s[2] * s[5]) / u[0];
    u
}

pub fn world(p: &[f32], c: &Config) -> [f32; 3] {
    let [a, e] = c.sensor_orientation;
    let [x, y, z] = [p[0], p[1], p[2]];
    let mut out = if a == 0.0 {
        let (s, co) = sincosd(e);
        [x, y * co + z * s, -y * s + z * co]
    } else {
        // The source uses double-precision trig then stores each entry as f32.
        #[allow(clippy::approx_constant)]
        let radians = 3.1415926_f64 / 180.0;
        let (se, ce) = ((e as f64) * radians).sin_cos();
        let (sa, ca) = ((a as f64) * radians).sin_cos();
        let (se, ce, sa, ca) = (se as f32, ce as f32, sa as f32, ca as f32);
        [
            ca * x + ce * sa * y + se * sa * z,
            -sa * x + ce * ca * y + se * ca * z,
            0.0 * x - se * y + ce * z,
        ]
    };
    out[2] += c.sensor_position[2];
    out
}

pub fn inside(p: &[f32; 3], boxes: &[f32; 12], count: u32) -> bool {
    boxes[..count as usize * 6]
        .chunks_exact(6)
        .any(|b| (0..3).all(|i| p[i] > b[2 * i] && p[i] < b[2 * i + 1]))
}

pub fn boresight(u: &[f32; 5], c: &Config) -> bool {
    ceiling(c)
        && c.boresight_filtering != 0
        && u[0] > 2.0
        && u[1].abs() < 6.0 * PI / 180.0
        && u[2].abs() < 6.0 * PI / 180.0
}

pub fn ceiling(c: &Config) -> bool {
    (90.0 - c.sensor_orientation[1]).abs() < 20.5
}

pub fn limits(r: f32, c: &Config) -> [f32; 4] {
    std::array::from_fn(|i| {
        let limit = c.gating_limits[i];
        if limit <= f32::MIN_POSITIVE {
            f32::MAX
        } else if i == 1 || i == 2 {
            atan((limit / 2.0) / r)
        } else {
            limit / 2.0
        }
    })
}

pub fn spread(r: f32, c: &Config) -> [f32; 4] {
    [
        if c.gating_limits[0] <= f32::MIN_POSITIVE {
            0.5
        } else {
            c.gating_limits[0]
        },
        if c.gating_limits[1] <= f32::MIN_POSITIVE {
            2.0 * PI / 180.0
        } else {
            2.0 * atan((c.gating_limits[1] / 2.0) / r)
        },
        // Deliberately retain TI's width here; height is used by limits().
        if c.gating_limits[2] <= f32::MIN_POSITIVE {
            2.0 * PI / 180.0
        } else {
            2.0 * atan((c.gating_limits[1] / 2.0) / r)
        },
        1.0,
    ]
}

pub fn unroll(max: f32, expected: f32, input: f32) -> f32 {
    let distance = expected - input;
    let factor = ((distance.abs() + max) / (2.0 * max)) as u16;
    if distance >= 0.0 {
        input + 2.0 * max * factor as f32
    } else {
        input - 2.0 * max * factor as f32
    }
}

pub fn distance(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let (_, ca) = sincosd((a[1] - b[1]) * (180.0 / PI));
    let (s1, c1) = sincosd(a[2] * (180.0 / PI));
    let (s2, c2) = sincosd(b[2] * (180.0 / PI));
    a[0] * a[0] + b[0] * b[0] - 2.0 * a[0] * b[0] * (c1 * c2 * ca + s1 * s2)
}

pub fn mahalanobis(v: &[f32; 4], d: &Matrix<4, 4>, size: usize) -> f32 {
    let mut result = 0.0;
    for j in 0..size {
        let mut inner = 0.0;
        for i in 0..size {
            inner += v[i] * d[i][j];
        }
        result += v[j] * inner;
    }
    result
}

pub fn jacobian(s: &[f32; 9]) -> Matrix<4, 9> {
    let [x, y, z, vx, vy, vz, ..] = *s;
    let r2 = x * x + y * y + z * z;
    let r = r2.sqrt();
    let xy2 = x * x + y * y;
    let xy = xy2.sqrt();
    [
        [x / r, y / r, z / r, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [y / xy2, -x / xy2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [
            -x * z / (r2 * xy),
            -y * z / (r2 * xy),
            xy / r2,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            (y * (vx * y - vy * x) + z * (vx * z - vz * x)) / (r2 * r),
            (x * (vy * x - vx * y) + z * (vy * z - vz * y)) / (r2 * r),
            (x * (vz * x - vx * z) + y * (vz * y - vy * z)) / (r2 * r),
            x / r,
            y / r,
            z / r,
            0.0,
            0.0,
            0.0,
        ],
    ]
}

pub fn motion(c: &Config) -> (Matrix<9, 9>, Matrix<9, 9>) {
    let dt = c.delta_t;
    let (dt2, dt3, dt4) = (dt.powf(2.0), dt.powf(3.0), dt.powf(4.0));
    let mut f = [[0.0; 9]; 9];
    let mut q = f;
    let base = [
        [dt4 / 4.0, dt3 / 2.0, dt2 / 2.0],
        [dt3 / 2.0, dt2, dt],
        [dt2 / 2.0, dt, 1.0],
    ];
    for axis in 0..3 {
        let variance = (0.5 * c.max_acceleration[axis]).powf(2.0);
        for row in 0..3 {
            f[row * 3 + axis][row * 3 + axis] = 1.0;
            for col in 0..3 {
                q[row * 3 + axis][col * 3 + axis] = base[row][col] * variance;
            }
        }
        f[axis][axis + 3] = dt;
        f[axis][axis + 6] = dt2 / 2.0;
        f[axis + 3][axis + 6] = dt;
    }
    (f, q)
}

// Explicit cofactors preserve the pinned kernel's arithmetic order.
mod inverse;
pub use inverse::inverse;
