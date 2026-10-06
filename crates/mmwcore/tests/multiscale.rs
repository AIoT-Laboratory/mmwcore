use mmwcore::tracking::multiscale::{ScatterBodyTracker, ScatterConfig};

#[test]
fn weak_target_initializes_and_empty_frames_release_it() {
    let mut tracker = ScatterBodyTracker::new(ScatterConfig::default()).unwrap();
    let points = [
        [2.0, 0.0, 1.0, 0.0, -15.0],
        [2.01, 0.0, 1.0, 0.0, -15.0],
        [1.99, 0.0, 1.0, 0.0, -15.0],
    ];
    assert!(tracker.step(&points, 0.1).unwrap().bodies.is_empty());
    assert!(tracker.step(&points, 0.1).unwrap().bodies.is_empty());
    let output = tracker.step(&points, 0.1).unwrap();
    assert_eq!(output.bodies.len(), 1);
    assert_eq!(output.bodies[0].component.id, 0);
    assert_eq!(output.bodies[0].body_measurement_members, [0, 1, 2]);
    for _ in 0..3 {
        assert!(tracker.step(&[], 0.1).unwrap().bodies[0].component.coasting);
    }
    assert!(tracker.step(&[], 0.1).unwrap().bodies.is_empty());
    assert!(tracker.state.tracks.is_empty());
}

#[test]
fn rejects_invalid_inputs_before_advancing_state() {
    let mut tracker = ScatterBodyTracker::new(ScatterConfig::default()).unwrap();
    assert!(tracker.step(&[], 0.0).is_err());
    assert!(tracker.step(&[[f64::NAN; 5]], 0.1).is_err());
    assert_eq!(tracker.state.time_s, 0.0);
    tracker.state.parents.insert(0, 0);
    assert!(tracker.step(&[], 0.1).is_err());
    assert_eq!(tracker.state.time_s, 0.0);
}
