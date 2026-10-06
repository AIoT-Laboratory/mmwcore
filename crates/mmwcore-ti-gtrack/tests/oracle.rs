use mmwcore_ti_gtrack::{Config, Engine};
use serde_json::Value;

fn config(tilt: f32) -> Config {
    Config {
        max_points: 16,
        max_tracks: 1,
        delta_t: 0.1,
        initial_velocity: 0.0,
        max_velocity: 4.0,
        velocity_resolution: 0.125,
        max_acceleration: [0.5; 3],
        boresight_filtering: 0,
        gating_gain: 4.0,
        gating_limits: [2.0; 4],
        allocation_snr: 1.0,
        allocation_obscured_snr: 1.0,
        allocation_velocity: 0.05,
        allocation_points: 4,
        allocation_distance: 0.8,
        allocation_max_velocity: 1.0,
        state_thresholds: [1, 1, 2, 4, 2, 6],
        sensor_position: [0.0, 0.0, 2.0],
        sensor_orientation: [0.0, tilt],
        boundary_count: 1,
        static_count: 1,
        occupancy_count: 1,
        boundary_boxes: [
            -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        static_boxes: [
            -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        occupancy_boxes: [
            -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        presence_points: 4,
        presence_on_to_off: 3,
        presence_velocity: 0.05,
    }
}

fn group(range: f64, velocity: f64) -> Vec<[f32; 5]> {
    (0..6)
        .map(|i| {
            let i = i as f64;
            [
                (range + 0.012 * i) as f32,
                (0.2 + 0.005 * i) as f32,
                (0.15 + 0.005 * i) as f32,
                if velocity == 0.0 {
                    0.0
                } else {
                    (velocity + 0.002 * i) as f32
                },
                (20.0 + i) as f32,
            ]
        })
        .collect()
}

fn frames() -> Vec<Vec<[f32; 5]>> {
    (0..6)
        .map(|k| group(2.0 + 0.03 * k as f64, 0.3))
        .chain((0..12).map(|_| group(2.2, 0.0)))
        .chain((0..60).map(|_| Vec::new()))
        .chain((0..6).map(|k| group(4.0 + 0.02 * k as f64, 0.2)))
        .chain((0..15).map(|_| Vec::new()))
        .collect()
}

fn compare(actual: &Value, expected: &Value, path: &str) {
    match expected {
        Value::Object(fields) => {
            for (key, value) in fields {
                compare(&actual[key], value, &format!("{path}.{key}"));
            }
        }
        Value::Array(values) => {
            let a = actual
                .as_array()
                .unwrap_or_else(|| panic!("{path}: expected array"));
            // Original C oracle represents matrices as flat arrays.
            let flattened: Vec<Value> = if values.first().is_some_and(Value::is_number) {
                a.iter()
                    .flat_map(|v| v.as_array().cloned().unwrap_or_else(|| vec![v.clone()]))
                    .collect()
            } else {
                a.clone()
            };
            assert_eq!(flattened.len(), values.len(), "{path}");
            for (i, (a, e)) in flattened.iter().zip(values).enumerate() {
                compare(a, e, &format!("{path}[{i}]"));
            }
        }
        Value::Number(value) => {
            if actual.is_i64() || actual.is_u64() {
                assert_eq!(actual, expected, "{path}");
                return;
            }
            let a = actual
                .as_f64()
                .unwrap_or_else(|| panic!("{path}: {actual}"));
            let e = value.as_f64().unwrap();
            assert!((a - e).abs() <= 2e-7 + 3e-6 * e.abs(), "{path}: {a} != {e}");
        }
        _ => assert_eq!(actual, expected, "{path}"),
    }
}

#[test]
fn oracle_comparison_distinguishes_counters_from_float_rounding() {
    compare(
        &serde_json::json!(0.0000001),
        &serde_json::json!(0),
        "float",
    );
    let changed_counter = std::panic::catch_unwind(|| {
        compare(
            &serde_json::json!(1_000_001),
            &serde_json::json!(1_000_000),
            "age",
        );
    });
    assert!(changed_counter.is_err());
}

#[test]
fn frozen_original_ti_3da_oracle() {
    let oracle: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/ti_gtrack_3da_oracle.json"
    ))
    .unwrap();
    for case in oracle["cases"].as_array().unwrap() {
        let tilt = case["tilt"].as_f64().unwrap() as f32;
        let variance = case["variance"].as_bool().unwrap();
        let mut engine = Engine::new(config(tilt)).unwrap();
        let inputs = frames();
        let expected_frames = case["frames"].as_array().unwrap();
        assert_eq!(inputs.len(), expected_frames.len(), "oracle frame count");
        for (i, (points, expected)) in inputs.iter().zip(expected_frames).enumerate() {
            let var = vec![[0.01, 0.001, 0.002, 0.02]; points.len()];
            let report = engine
                .step(points, variance.then_some(var.as_slice()))
                .unwrap();
            let actual = serde_json::to_value(report).unwrap();
            compare(
                &actual,
                expected,
                &format!("tilt={tilt},variance={variance},frame={i}"),
            );
        }
    }
}

/// Explicit development check; production/default tests need no external library.
#[cfg(feature = "reference-plugin")]
#[test]
fn c_bridge_static_multi_target_turns_and_velocity_aliases() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../build/ti-gtrack/manifest.json");
    for tilt in [0.0, 30.0, 90.0] {
        let mut c = config(tilt);
        c.max_points = 64;
        c.max_tracks = 4;
        c.sensor_orientation[0] = if tilt == 30.0 { 12.0 } else { 0.0 };
        for variance in [false, true] {
            let mut native = Engine::new(c.clone()).unwrap();
            let mut reference = Engine::load(&manifest, c.clone()).unwrap();
            for k in 0..240 {
                let mut points = Vec::new();
                for person in 0..2 {
                    let phase = k % 80;
                    if phase > 60 {
                        continue;
                    }
                    let velocity = if phase < 16 {
                        0.3
                    } else if phase < 40 {
                        0.0
                    } else {
                        -0.3
                    };
                    if person == 0 && (16..40).contains(&phase) {
                        continue;
                    }
                    let r = 2.0 + person as f64 + 0.01 * phase.min(40) as f64;
                    for mut p in group(r, velocity) {
                        p[1] += if person == 0 { -0.4 } else { 0.5 };
                        if phase >= 40 {
                            p[3] += 8.0;
                        }
                        points.push(p);
                    }
                }
                let prefix = points.len();
                if (16..40).contains(&(k % 80)) {
                    points.push([2.2, -0.18, 0.16, 0.0, 1.0]);
                }
                let var = vec![[0.01, 0.001, 0.002, 0.02]; points.len()];
                let vars = variance.then_some(var.as_slice());
                let a = native
                    .step_with_static(&points, vars, Some(prefix))
                    .unwrap();
                let b = reference
                    .step_with_static(&points, vars, Some(prefix))
                    .unwrap();
                assert_eq!(a.point_uid, b.point_uid, "tilt={tilt}, frame={k}");
                assert_eq!(a.point_unique, b.point_unique);
                assert_eq!(a.point_static, b.point_static);
                let mut expected = serde_json::to_value(b).unwrap();
                expected.as_object_mut().unwrap().remove("benchmark_ticks");
                compare(
                    &serde_json::to_value(a).unwrap(),
                    &expected,
                    &format!("tilt={tilt}, variance={variance}, frame={k}"),
                );
            }
        }
    }
}
