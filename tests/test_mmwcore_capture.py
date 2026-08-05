from __future__ import annotations

import numpy as np
import pytest

from mmwcore.core import RawADCFrame
from mmwcore.io import PacketLossStats
from mmwcore.session import RadarCaptureResult, RadarOnlyCaptureSession, save_raw_adc_capture


def test_radar_only_capture_session_prepares_and_captures_frame() -> None:
    events: list[str] = []
    radar = FakeRadar(events)
    recorder = FakeRecorder(events)
    reader = FakeFrameReader(events)
    session = RadarOnlyCaptureSession(radar=radar, recorder=recorder, frame_reader=reader)

    with session:
        result = session.capture_frame(
            frame_id="frame-0",
            radar_config_lines=["", "% comment", "sensorStop", "profileCfg 0"],
        )

    assert result.loaded_commands == ("sensorStop", "profileCfg 0")
    assert result.raw.frame_id == "frame-0"
    np.testing.assert_array_equal(result.raw.samples, np.array([1, 2, 3, 4], dtype=np.int16))
    assert result.packet_stats.missing_count == 0
    assert events == [
        "radar.open",
        "reader.open",
        "radar.load_config_lines",
        "recorder.configure_fpga",
        "recorder.start_record",
        "radar.sensor_start",
        "reader.read_frame",
        "radar.sensor_stop",
        "recorder.stop_record",
        "reader.close",
        "radar.close",
    ]


def test_radar_only_capture_session_skips_manual_start_when_config_includes_start() -> None:
    events: list[str] = []
    session = RadarOnlyCaptureSession(
        radar=FakeRadar(events),
        recorder=FakeRecorder(events),
        frame_reader=FakeFrameReader(events),
    )

    session.open()
    result = session.capture_frame(
        radar_config_lines=["sensorStart"],
        include_sensor_start=True,
    )
    session.close()

    assert result.loaded_commands == ("sensorStart",)
    assert "radar.sensor_start" not in events


def test_radar_only_capture_session_stops_recording_when_read_fails() -> None:
    events: list[str] = []
    session = RadarOnlyCaptureSession(
        radar=FakeRadar(events),
        recorder=FakeRecorder(events),
        frame_reader=FakeFrameReader(events, fail=True),
    )

    session.open()
    with pytest.raises(RuntimeError, match="read failed"):
        session.capture_frame()

    assert events[-2:] == ["radar.sensor_stop", "recorder.stop_record"]


def test_save_raw_adc_capture_writes_binary_and_artifact(tmp_path) -> None:
    raw = RawADCFrame(
        np.array([1, 2, 3, 4], dtype=np.int16),
        frame_id="frame-0",
        source="fixture",
        metadata={"packet_loss": 0},
    )
    result = RadarCaptureResult(
        raw=raw,
        packet_stats=PacketLossStats(
            expected_packets=1,
            received_packets=1,
            missing_packet_numbers=(),
            duplicate_packet_numbers=(),
        ),
    )

    artifact = save_raw_adc_capture(
        result,
        tmp_path / "raw" / "sample.bin",
        sample_id="sample",
        artifact_manifest=tmp_path / "artifacts" / "raw_adc.jsonl",
        root=tmp_path,
        metadata={"adc_spec": {"num_chirps": 2}},
    )

    assert artifact is not None
    np.testing.assert_array_equal(
        np.fromfile(tmp_path / "raw" / "sample.bin", dtype=np.int16),
        np.array([1, 2, 3, 4], dtype=np.int16),
    )
    assert artifact.adc == "raw/sample.bin"
    assert artifact.metadata["adc_spec"] == {"num_chirps": 2}


def test_save_raw_adc_capture_indexes_ti_cfg_metadata(tmp_path) -> None:
    cfg_path = tmp_path / "radar.cfg"
    cfg_path.write_text(
        "\n".join(
            [
                "channelCfg 15 1 0",
                "profileCfg 0 60 30 7 57.14 0 0 60 1 128 5209 0 0 158",
                "chirpCfg 0 0 0 0 0 0 0 1",
                "frameCfg 0 0 2 0 10 1 0",
            ]
        ),
        encoding="utf-8",
    )
    raw = RawADCFrame(np.array([1, 2, 3, 4], dtype=np.int16), frame_id="frame-0")
    result = RadarCaptureResult(
        raw=raw,
        packet_stats=PacketLossStats(
            expected_packets=1,
            received_packets=1,
            missing_packet_numbers=(),
            duplicate_packet_numbers=(),
        ),
    )

    artifact = save_raw_adc_capture(
        result,
        tmp_path / "raw" / "sample.bin",
        sample_id="sample",
        artifact_manifest=tmp_path / "artifacts" / "raw_adc.jsonl",
        root=tmp_path,
        ti_cfg=cfg_path,
        metadata={"operator": "unit"},
    )

    assert artifact is not None
    assert artifact.metadata["ti_cfg"] == str(cfg_path)
    assert artifact.metadata["operator"] == "unit"
    assert artifact.metadata["adc_spec"] == {
        "num_chirps": 2,
        "num_rx": 4,
        "num_samples": 128,
        "layout": "iq_interleaved",
    }


class FakeRadar:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def open(self) -> object:
        self.events.append("radar.open")
        return self

    def close(self) -> None:
        self.events.append("radar.close")

    def load_config_lines(self, lines, *, include_sensor_start=False):
        self.events.append("radar.load_config_lines")
        commands = []
        for line in lines:
            command = line.strip()
            if not command or command.startswith("%"):
                continue
            if command == "sensorStart" and not include_sensor_start:
                continue
            commands.append(command)
        return tuple(commands)

    def sensor_start(self) -> str:
        self.events.append("radar.sensor_start")
        return "sensorStart"

    def sensor_stop(self) -> str:
        self.events.append("radar.sensor_stop")
        return "sensorStop"


class FakeRecorder:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def configure_fpga(self) -> object:
        self.events.append("recorder.configure_fpga")
        return None

    def start_record(self) -> object:
        self.events.append("recorder.start_record")
        return None

    def stop_record(self) -> object:
        self.events.append("recorder.stop_record")
        return None


class FakeFrameReader:
    def __init__(self, events: list[str], *, fail: bool = False) -> None:
        self.events = events
        self.fail = fail

    def open(self) -> object:
        self.events.append("reader.open")
        return self

    def close(self) -> None:
        self.events.append("reader.close")

    def read_frame(self, *, frame_id=None):
        self.events.append("reader.read_frame")
        if self.fail:
            raise RuntimeError("read failed")
        return (
            RawADCFrame(np.array([1, 2, 3, 4], dtype=np.int16), frame_id=frame_id),
            PacketLossStats(
                expected_packets=1,
                received_packets=1,
                missing_packet_numbers=(),
                duplicate_packet_numbers=(),
            ),
        )
