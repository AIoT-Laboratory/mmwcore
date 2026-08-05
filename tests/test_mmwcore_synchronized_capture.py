from __future__ import annotations

import json
from collections.abc import Callable, Iterator

import pytest

from mmwcore.config import RadarCaptureSpec, RadarProfile
from mmwcore.core import ADCComplexLayout, ADCFrameSpec
from mmwcore.session import (
    CAPTURE_SYNC_CONTROL_VERSION,
    CaptureSyncEvent,
    CaptureSyncEventKind,
    CaptureSyncEventWriter,
    RadarCameraMatchStatus,
    RadarCaptureTiming,
    RadarFrameTriggerMode,
    SoftwareCaptureState,
    SoftwareSynchronizedCapture,
    SyncControlAction,
    SyncControlMessage,
    build_causal_radar_camera_matches,
    export_radar_camera_alignment,
    inspect_synchronized_capture,
    load_capture_sync_events,
    load_radar_camera_alignment,
    validate_capture_session,
)


def test_sync_control_message_round_trips() -> None:
    message = SyncControlMessage(
        capture_id="subject-01_20260728",
        action=SyncControlAction.RADAR_START,
        sequence=3,
        metadata={"source": "test"},
    )

    assert SyncControlMessage.parse(message.encode()) == message


def test_sync_control_message_accepts_lua_arm_payload() -> None:
    payload = json.dumps(
        {
            "action": "arm",
            "capture_id": "capture-001",
            "metadata": {
                "radar": {"capture_device": "DCA1000", "sensor": "xwr68xx"},
                "radar_capture": _capture_spec(
                    num_frames=600,
                    periodicity_ms=100.0,
                ).to_record(),
                "radar_timing": _timing(
                    num_frames=600,
                    periodicity_ms=100.0,
                ).to_record(),
                "session": _session(),
            },
            "sequence": 0,
            "version": CAPTURE_SYNC_CONTROL_VERSION,
        }
    ).encode()

    message = SyncControlMessage.parse(payload)

    assert message.action is SyncControlAction.ARM
    timing = RadarCaptureTiming.from_record(message.metadata["radar_timing"])
    assert timing.num_frames == 600
    assert timing.expected_duration_s == pytest.approx(60.0)
    assert RadarCaptureSpec.from_record(message.metadata["radar_capture"]).num_frames == 600


def test_capture_session_accepts_null_empty_scene() -> None:
    session = _session()
    session.update(action="null", expected_people=0)

    assert validate_capture_session(session) == session


@pytest.mark.parametrize(
    ("action", "expected_people"),
    [("null", 1), ("stand", 0)],
)
def test_capture_session_rejects_inconsistent_empty_scene(
    action: str,
    expected_people: int,
) -> None:
    session = _session()
    session.update(action=action, expected_people=expected_people)

    with pytest.raises(ValueError, match="expected_people 0 exactly"):
        validate_capture_session(session)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"version":"OPENMMW_SYNC_V0"}',
        b'{"version":"OPENMMW_SYNC_V1","capture_id":"capture","action":"arm","sequence":0}',
        b'{"version":"OPENMMW_SYNC_V2","capture_id":"../capture","action":"arm","sequence":0}',
        b'{"version":"OPENMMW_SYNC_V2","capture_id":"capture","action":"unknown","sequence":0}',
        b'{"version":"OPENMMW_SYNC_V2","capture_id":"capture","action":"arm","sequence":-1}',
        b"\xff",
    ],
)
def test_sync_control_message_rejects_invalid_payload(payload: bytes) -> None:
    with pytest.raises(ValueError):
        SyncControlMessage.parse(payload)


def test_capture_sync_event_writer_uses_contiguous_indices(tmp_path) -> None:
    monotonic = iter([100, 200])
    utc = iter([1_000, 1_100])
    path = tmp_path / "events.jsonl"

    with CaptureSyncEventWriter(
        path,
        capture_id="capture",
        monotonic_clock=_clock(monotonic),
        utc_clock=_clock(utc),
    ) as writer:
        writer.append(CaptureSyncEventKind.CAMERA_ARMED, control_sequence=0)
        writer.append(CaptureSyncEventKind.RADAR_START, control_sequence=1)

    events = load_capture_sync_events(path)
    assert [event.event_index for event in events] == [0, 1]
    assert [event.monotonic_ns for event in events] == [100, 200]


def test_software_capture_writes_auditable_artifact(tmp_path) -> None:
    monotonic = iter([310_000_000])
    utc = iter([1_310_000_000])
    capture = SoftwareSynchronizedCapture(
        tmp_path,
        capture_id="capture-001",
        camera={"device": 0, "fps": 30.0},
        radar={"sensor": "xwr68xx", "capture_device": "DCA1000"},
        radar_capture=_capture_spec(num_frames=1, periodicity_ms=100.0),
        radar_timing=_timing(num_frames=1, periodicity_ms=100.0),
        session=_session(),
        monotonic_clock=_clock(monotonic),
        utc_clock=_clock(utc),
    )

    capture.arm(
        control_sequence=0,
        monotonic_ns=100_000_000,
        utc_ns=1_100_000_000,
    )
    pre_roll = capture.camera_frames_root / "frame_000000.jpg"
    pre_roll.write_bytes(b"jpeg")
    capture.record_camera_frame(
        pre_roll.name,
        width=640,
        height=480,
        monotonic_ns=180_000_000,
        utc_ns=1_180_000_000,
    )
    capture.mark_radar_start(
        control_sequence=1,
        monotonic_ns=200_000_000,
        utc_ns=1_200_000_000,
    )
    for index in range(3):
        path = capture.camera_frames_root / f"frame_{index + 1:06d}.jpg"
        path.write_bytes(b"jpeg")
        capture.record_camera_frame(
            path.name,
            width=640,
            height=480,
            monotonic_ns=210_000_000 + index * 30_000_000,
            utc_ns=1_210_000_000 + index * 30_000_000,
        )
    _write_adc(capture)
    capture.mark_radar_stop(
        control_sequence=2,
        monotonic_ns=300_000_000,
        utc_ns=1_300_000_000,
    )
    manifest = capture.finish(control_sequence=3)

    assert capture.state is SoftwareCaptureState.CLOSED
    inspection = inspect_synchronized_capture(manifest)
    assert inspection.capture_id == "capture-001"
    assert inspection.camera_frame_count == 4
    assert inspection.frames_before_radar == 1
    assert inspection.frames_during_radar == 3
    assert inspection.radar_duration_s == pytest.approx(0.1)
    assert inspection.expected_radar_duration_s == pytest.approx(0.1)
    assert inspection.radar_duration_error_ms == pytest.approx(0.0)
    assert inspection.causal_match_count == 1
    assert inspection.causal_unmatched_frame_count == 0
    assert inspection.causal_frame_lag_ms_median == pytest.approx(20.0)
    assert inspection.frame_interval_ms_median == pytest.approx(30.0)
    assert inspection.radar_adc_exists is True
    assert inspection.radar_adc_complete is True
    assert inspection.radar_adc_size_bytes == 4
    assert inspection.session["action"] == "stand"
    assert (
        inspection.synchronization_claim
        == "same_host_software_timestamped_not_hardware_synchronized"
    )


def test_causal_alignment_rejects_equal_and_future_camera_frames() -> None:
    events = (
        _event(CaptureSyncEventKind.CAMERA_ARMED, 0, 0),
        _event(CaptureSyncEventKind.CAMERA_FRAME, 1, 90_000_000, frame_index=0),
        _event(CaptureSyncEventKind.CAMERA_FRAME, 2, 100_000_000, frame_index=1),
        _event(CaptureSyncEventKind.RADAR_START, 3, 100_000_000),
        _event(CaptureSyncEventKind.CAMERA_FRAME, 4, 180_000_000, frame_index=2),
        _event(CaptureSyncEventKind.CAMERA_FRAME, 5, 210_000_000, frame_index=3),
        _event(CaptureSyncEventKind.CAMERA_FRAME, 6, 260_000_000, frame_index=4),
    )

    matches = build_causal_radar_camera_matches(
        events,
        _timing(num_frames=3, periodicity_ms=100.0),
        max_lag_ms=30.0,
    )

    assert [match.camera_frame_index for match in matches] == [0, 2, 4]
    assert [match.lag_ms for match in matches] == [10.0, 20.0, 40.0]
    assert [match.status for match in matches] == [
        RadarCameraMatchStatus.MATCHED,
        RadarCameraMatchStatus.MATCHED,
        RadarCameraMatchStatus.LAG_EXCEEDS_LIMIT,
    ]
    assert all(
        match.camera_monotonic_ns is None or match.camera_monotonic_ns < match.radar_monotonic_ns
        for match in matches
    )


def test_causal_alignment_marks_missing_preroll_explicitly() -> None:
    events = (
        _event(CaptureSyncEventKind.CAMERA_ARMED, 0, 0),
        _event(CaptureSyncEventKind.RADAR_START, 1, 100_000_000),
        _event(CaptureSyncEventKind.CAMERA_FRAME, 2, 110_000_000, frame_index=0),
    )

    matches = build_causal_radar_camera_matches(
        events,
        _timing(num_frames=1, periodicity_ms=100.0),
    )

    assert matches[0].status is RadarCameraMatchStatus.NO_CAUSAL_FRAME
    assert matches[0].camera_frame_index is None


def test_radar_camera_alignment_artifact_round_trips_and_detects_tampering(
    tmp_path,
) -> None:
    capture = _completed_capture(tmp_path)
    output_root = tmp_path / "alignment"
    exported = export_radar_camera_alignment(
        capture.capture_root,
        output_root,
        max_lag_ms=60.0,
    )

    artifact = load_radar_camera_alignment(output_root)

    assert artifact.capture_id == "capture-alignment"
    assert artifact.radar_capture == _capture_spec()
    assert artifact.session == _session()
    assert artifact.matches[0].status is RadarCameraMatchStatus.MATCHED
    assert artifact.matches[0].camera_frame_index == 0
    manifest = json.loads(exported.manifest_path.read_text(encoding="utf-8"))
    assert artifact.index_sha256 == manifest["index_sha256"]

    exported.index_path.write_text(
        exported.index_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="index digest"):
        load_radar_camera_alignment(output_root)


def test_software_capture_rejects_existing_capture_id(tmp_path) -> None:
    (tmp_path / "capture").mkdir()
    capture = SoftwareSynchronizedCapture(
        tmp_path,
        capture_id="capture",
        camera={"device": 0},
        radar={"sensor": "xwr68xx"},
        radar_capture=_capture_spec(),
        radar_timing=_timing(),
        session=_session(),
    )

    with pytest.raises(FileExistsError):
        capture.arm(control_sequence=0)


def test_software_capture_rejects_radar_path_escape(tmp_path) -> None:
    with pytest.raises(ValueError, match="relative"):
        SoftwareSynchronizedCapture(
            tmp_path,
            capture_id="capture",
            camera={"device": 0},
            radar={"sensor": "xwr68xx"},
            radar_capture=_capture_spec(),
            radar_timing=_timing(),
            radar_adc="../adc.bin",
            session=_session(),
        )


def test_software_capture_requires_stop_before_finalization(tmp_path) -> None:
    capture = _recording_capture(tmp_path, capture_id="capture-state")
    _write_adc(capture)

    with pytest.raises(RuntimeError, match="expected stopped"):
        capture.finish(control_sequence=2)


def test_software_capture_rejects_incomplete_adc_after_stop(tmp_path) -> None:
    capture = _recording_capture(tmp_path, capture_id="capture-size")
    (capture.capture_root / "radar" / "adc_data.bin").write_bytes(b"bad")
    capture.mark_radar_stop(control_sequence=2)

    with pytest.raises(ValueError, match="does not match"):
        capture.finish(control_sequence=3)

    assert capture.state is SoftwareCaptureState.STOPPED


def test_sync_inspection_rejects_missing_camera_frame(tmp_path) -> None:
    monotonic = iter([100, 200, 300, 400])
    utc = iter([1_000, 2_000, 3_000, 4_000])
    capture = SoftwareSynchronizedCapture(
        tmp_path,
        capture_id="capture",
        camera={"device": 0},
        radar={"sensor": "xwr68xx"},
        radar_capture=_capture_spec(),
        radar_timing=_timing(),
        session=_session(),
        monotonic_clock=_clock(monotonic),
        utc_clock=_clock(utc),
    )
    capture.arm(control_sequence=0)
    capture.mark_radar_start(control_sequence=1)
    frame = capture.camera_frames_root / "frame_000000.jpg"
    frame.write_bytes(b"jpeg")
    capture.record_camera_frame(
        frame.name,
        width=640,
        height=480,
        monotonic_ns=250,
        utc_ns=2_500,
    )
    _write_adc(capture)
    capture.mark_radar_stop(control_sequence=2)
    manifest = capture.finish(control_sequence=3)
    frame.unlink()

    with pytest.raises(ValueError, match="does not exist"):
        inspect_synchronized_capture(manifest)


def test_sync_inspection_rejects_adc_changed_after_publication(tmp_path) -> None:
    capture = _completed_capture(tmp_path)
    adc_path = capture.capture_root / "radar" / "adc_data.bin"
    with adc_path.open("ab") as stream:
        stream.write(b"\0")

    with pytest.raises(ValueError, match="no longer matches"):
        inspect_synchronized_capture(capture.capture_root)


def test_synchronized_capture_manifest_records_uncertainty(tmp_path) -> None:
    capture = SoftwareSynchronizedCapture(
        tmp_path,
        capture_id="capture",
        camera={"device": 0},
        radar={"sensor": "xwr68xx"},
        radar_capture=_capture_spec(),
        radar_timing=_timing(),
        session=_session(),
    )
    capture.arm(control_sequence=0)
    capture.mark_radar_start(control_sequence=1)
    frame = capture.camera_frames_root / "frame_000000.jpg"
    frame.write_bytes(b"jpeg")
    capture.record_camera_frame(
        frame.name,
        width=640,
        height=480,
        monotonic_ns=200,
        utc_ns=2_000,
    )
    _write_adc(capture)
    capture.mark_radar_stop(control_sequence=2, monotonic_ns=300, utc_ns=3_000)
    manifest = capture.finish(control_sequence=3)

    record = json.loads(manifest.read_text(encoding="utf-8"))
    assert record["synchronization_mode"] == "software_timestamped"
    assert record["session"] == _session()
    assert "session" not in record["metadata"]
    assert record["metadata"]["camera_recording_window"] == ("camera_armed_through_radar_stop")
    assert record["metadata"]["radar_start_marker"] == ("lower_bound_before_ar1_start_frame")
    assert "camera_exposure_and_driver_buffering" in record["metadata"]["known_uncertainty"]


def _clock(values: Iterator[int]) -> Callable[[], int]:
    return lambda: next(values)


def _recording_capture(tmp_path, *, capture_id: str) -> SoftwareSynchronizedCapture:
    capture = SoftwareSynchronizedCapture(
        tmp_path,
        capture_id=capture_id,
        camera={"device": 0},
        radar={"sensor": "xwr68xx"},
        radar_capture=_capture_spec(),
        radar_timing=_timing(),
        session=_session(),
    )
    capture.arm(control_sequence=0)
    capture.mark_radar_start(control_sequence=1)
    frame = capture.camera_frames_root / "frame_000000.jpg"
    frame.write_bytes(b"jpeg")
    capture.record_camera_frame(
        frame.name,
        width=640,
        height=480,
        monotonic_ns=200,
        utc_ns=2_000,
    )
    return capture


def _completed_capture(tmp_path) -> SoftwareSynchronizedCapture:
    capture = SoftwareSynchronizedCapture(
        tmp_path,
        capture_id="capture-alignment",
        camera={"device": 0, "fps": 30.0},
        radar={"sensor": "xwr68xx", "capture_device": "DCA1000"},
        radar_capture=_capture_spec(),
        radar_timing=_timing(),
        session=_session(),
    )
    capture.arm(
        control_sequence=0,
        monotonic_ns=100_000_000,
        utc_ns=1_100_000_000,
    )
    frame = capture.camera_frames_root / "frame_000000.jpg"
    frame.write_bytes(b"jpeg")
    capture.record_camera_frame(
        frame.name,
        width=640,
        height=480,
        monotonic_ns=190_000_000,
        utc_ns=1_190_000_000,
    )
    capture.mark_radar_start(
        control_sequence=1,
        monotonic_ns=200_000_000,
        utc_ns=1_200_000_000,
    )
    frame = capture.camera_frames_root / "frame_000001.jpg"
    frame.write_bytes(b"jpeg")
    capture.record_camera_frame(
        frame.name,
        width=640,
        height=480,
        monotonic_ns=250_000_000,
        utc_ns=1_250_000_000,
    )
    _write_adc(capture)
    capture.mark_radar_stop(
        control_sequence=2,
        monotonic_ns=300_000_000,
        utc_ns=1_300_000_000,
    )
    capture.finish(control_sequence=3)
    return capture


def _event(
    kind: CaptureSyncEventKind,
    event_index: int,
    monotonic_ns: int,
    *,
    frame_index: int | None = None,
) -> CaptureSyncEvent:
    metadata = (
        {
            "frame_index": frame_index,
            "path": f"frame_{frame_index:06d}.jpg",
        }
        if frame_index is not None
        else {}
    )
    return CaptureSyncEvent(
        capture_id="capture",
        kind=kind,
        event_index=event_index,
        monotonic_ns=monotonic_ns,
        utc_ns=monotonic_ns,
        metadata=metadata,
    )


def _timing(
    *,
    num_frames: int = 1,
    periodicity_ms: float = 100.0,
) -> RadarCaptureTiming:
    return RadarCaptureTiming(
        num_frames=num_frames,
        frame_periodicity_ms=periodicity_ms,
        frame_trigger_mode=RadarFrameTriggerMode.SOFTWARE,
    )


def _capture_spec(
    *,
    num_frames: int = 1,
    periodicity_ms: float = 100.0,
) -> RadarCaptureSpec:
    profile = RadarProfile(
        num_adc_samples=1,
        num_chirps_per_tx=1,
        num_tx=1,
        num_rx=1,
    )
    return RadarCaptureSpec(
        profile=profile,
        adc=ADCFrameSpec(
            num_chirps=1,
            num_rx=1,
            num_samples=1,
            layout=ADCComplexLayout.GROUP2_I_THEN_Q,
        ),
        tx_order=(0,),
        frame_periodicity_s=periodicity_ms / 1_000.0,
        num_frames=num_frames,
    )


def _session() -> dict[str, object]:
    return {
        "subject_id": "subject-001",
        "scene_id": "scene-001",
        "action": "stand",
        "take_index": 0,
        "expected_people": 1,
        "radar_mount": {
            "height_m": 1.2,
            "yaw_deg": 0.0,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
        },
        "detection_region": {
            "coordinate_frame": "mount_compensated_forward_lateral_up",
            "center_xyz_m": [1.25, 0.125, 1.1],
            "length_m": 1.5,
            "width_m": 1.25,
            "height_m": 2.2,
        },
    }


def _write_adc(capture: SoftwareSynchronizedCapture) -> None:
    expected_size = capture.radar_capture.expected_size_bytes
    assert expected_size is not None
    (capture.capture_root / "radar" / "adc_data.bin").write_bytes(b"\0" * expected_size)
