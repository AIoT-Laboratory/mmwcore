"""Stateful same-host software synchronization for radar-camera capture."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from mmwcore._compat import StrEnum
from mmwcore.config import RadarCaptureSpec

from .sync_protocol import (
    CAPTURE_SYNC_CONTROL_VERSION,
    CaptureSyncEventKind,
    CaptureSyncEventWriter,
    validate_capture_id,
)
from .synchronized_capture import (
    RadarCaptureTiming,
    SynchronizationMode,
    SynchronizedCaptureArtifact,
    validate_capture_session,
    write_synchronized_capture_manifest,
)


class SoftwareCaptureState(StrEnum):
    """Lifecycle states for one software-timestamped capture."""

    NEW = "new"
    ARMED = "armed"
    RECORDING = "recording"
    STOPPED = "stopped"
    CLOSED = "closed"
    ABORTED = "aborted"


class SoftwareSynchronizedCapture:
    """Own one immutable camera/event artifact beside a DCA1000 ADC file.

    Radar control events and camera frames are timestamped by this process.
    The radar-start marker represents receipt of a control message sent before
    ``ar1.StartFrame()``; it is not a measurement of the physical RF trigger.
    """

    def __init__(
        self,
        output_root: str | Path,
        *,
        capture_id: str,
        camera: Mapping[str, Any],
        radar: Mapping[str, Any],
        radar_capture: RadarCaptureSpec,
        radar_timing: RadarCaptureTiming,
        session: Mapping[str, Any],
        radar_adc: str = "radar/adc_data.bin",
        metadata: Mapping[str, Any] | None = None,
        monotonic_clock: Callable[[], int] = time.monotonic_ns,
        utc_clock: Callable[[], int] = time.time_ns,
    ) -> None:
        validate_capture_id(capture_id)
        self.output_root = Path(output_root)
        self.capture_id = capture_id
        self.camera = dict(camera)
        self.radar = dict(radar)
        self.radar_capture = radar_capture
        self.radar_timing = radar_timing
        self.session = validate_capture_session(dict(session))
        self.radar_adc = radar_adc
        self.metadata = dict(metadata or {})
        json.dumps(self.camera)
        json.dumps(self.radar)
        json.dumps(self.metadata)
        if not isinstance(radar_capture, RadarCaptureSpec):
            raise TypeError("radar_capture must be a RadarCaptureSpec.")
        if not isinstance(radar_timing, RadarCaptureTiming):
            raise TypeError("radar_timing must be a RadarCaptureTiming.")
        if (
            radar_capture.num_frames != radar_timing.num_frames
            or radar_capture.frame_periodicity_s is None
            or not math.isclose(
                radar_capture.frame_periodicity_s * 1_000.0,
                radar_timing.frame_periodicity_ms,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ValueError("Radar capture spec and synchronized frame timing do not match.")
        self.capture_root = self.output_root / capture_id
        self.camera_frames_root = self.capture_root / "camera"
        self.manifest_path = self.capture_root / "manifest.json"
        self.event_log_path = self.capture_root / "events.jsonl"
        self._radar_adc_path = self._resolve_capture_reference(radar_adc)
        self._writer = CaptureSyncEventWriter(
            self.event_log_path,
            capture_id=capture_id,
            monotonic_clock=monotonic_clock,
            utc_clock=utc_clock,
        )
        self._state = SoftwareCaptureState.NEW
        self._frame_count = 0

    @property
    def state(self) -> SoftwareCaptureState:
        return self._state

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def is_recording(self) -> bool:
        return self._state is SoftwareCaptureState.RECORDING

    @property
    def accepts_camera_frames(self) -> bool:
        """Whether camera frames belong to the armed capture.

        Frames captured while armed form the causal pre-roll required to align
        the first radar frame without selecting a future image.
        """

        return self._state in {
            SoftwareCaptureState.ARMED,
            SoftwareCaptureState.RECORDING,
        }

    def arm(
        self,
        *,
        control_sequence: int,
        monotonic_ns: int | None = None,
        utc_ns: int | None = None,
    ) -> None:
        """Create a new exclusive capture directory and arm the camera."""

        self._require_state(SoftwareCaptureState.NEW)
        self.capture_root.mkdir(parents=True, exist_ok=False)
        self.camera_frames_root.mkdir()
        self._radar_adc_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer.open()
        try:
            self._writer.append(
                CaptureSyncEventKind.CAMERA_ARMED,
                control_sequence=control_sequence,
                metadata={"camera": self.camera},
                monotonic_ns=monotonic_ns,
                utc_ns=utc_ns,
            )
        except BaseException:
            self._writer.close()
            raise
        self._state = SoftwareCaptureState.ARMED

    def mark_radar_start(
        self,
        *,
        control_sequence: int,
        monotonic_ns: int | None = None,
        utc_ns: int | None = None,
    ) -> None:
        """Record receipt of the marker sent immediately before StartFrame."""

        self._require_state(SoftwareCaptureState.ARMED)
        self._writer.append(
            CaptureSyncEventKind.RADAR_START,
            control_sequence=control_sequence,
            metadata={
                "semantics": "control_received_before_ar1_start_frame",
            },
            monotonic_ns=monotonic_ns,
            utc_ns=utc_ns,
        )
        self._state = SoftwareCaptureState.RECORDING

    def record_camera_frame(
        self,
        relative_path: str,
        *,
        width: int,
        height: int,
        monotonic_ns: int,
        utc_ns: int,
    ) -> None:
        """Record one already-persisted frame in the shared event clock."""

        self._require_states(
            SoftwareCaptureState.ARMED,
            SoftwareCaptureState.RECORDING,
        )
        frame_path = self._resolve_camera_frame(relative_path)
        if not frame_path.is_file():
            raise FileNotFoundError(f"Camera frame was not persisted: {frame_path}")
        for name, value in (("width", width), ("height", height)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"Camera frame {name} must be a positive integer.")
        self._writer.append(
            CaptureSyncEventKind.CAMERA_FRAME,
            metadata={
                "frame_index": self._frame_count,
                "path": Path(relative_path).as_posix(),
                "width": width,
                "height": height,
            },
            monotonic_ns=monotonic_ns,
            utc_ns=utc_ns,
        )
        self._frame_count += 1

    def mark_radar_stop(
        self,
        *,
        control_sequence: int,
        monotonic_ns: int | None = None,
        utc_ns: int | None = None,
    ) -> None:
        """Record radar completion before the capture device flushes its file."""

        self._require_state(SoftwareCaptureState.RECORDING)
        self._writer.append(
            CaptureSyncEventKind.RADAR_STOP,
            control_sequence=control_sequence,
            metadata={"semantics": "control_received_after_radar_capture_wait"},
            monotonic_ns=monotonic_ns,
            utc_ns=utc_ns,
        )
        self._state = SoftwareCaptureState.STOPPED

    def finish(
        self,
        *,
        control_sequence: int,
    ) -> Path:
        """Verify the flushed ADC file and publish the immutable manifest."""

        self._require_state(SoftwareCaptureState.STOPPED)
        if self._frame_count == 0:
            raise ValueError("Cannot finish a synchronized capture without camera frames.")
        self._validate_radar_adc()
        self._writer.append(
            CaptureSyncEventKind.CAPTURE_CLOSED,
            control_sequence=control_sequence,
            metadata={"semantics": "adc_file_flushed_and_size_verified"},
        )
        self._writer.close()
        artifact = SynchronizedCaptureArtifact(
            capture_id=self.capture_id,
            synchronization_mode=SynchronizationMode.SOFTWARE_TIMESTAMPED,
            event_log=self.event_log_path.name,
            camera_frames=self.camera_frames_root.name,
            camera_frame_count=self._frame_count,
            camera=self.camera,
            radar=self.radar,
            radar_capture=self.radar_capture,
            radar_timing=self.radar_timing,
            session=self.session,
            radar_adc=Path(self.radar_adc).as_posix(),
            metadata={
                "control_protocol": CAPTURE_SYNC_CONTROL_VERSION,
                "camera_recording_window": "camera_armed_through_radar_stop",
                "radar_start_marker": "lower_bound_before_ar1_start_frame",
                "known_uncertainty": [
                    "udp_delivery",
                    "lua_scheduling",
                    "ar1_start_frame_api",
                    "camera_exposure_and_driver_buffering",
                ],
                **self.metadata,
            },
        )
        try:
            output = write_synchronized_capture_manifest(self.manifest_path, artifact)
        except BaseException:
            self._state = SoftwareCaptureState.ABORTED
            raise
        self._state = SoftwareCaptureState.CLOSED
        return output

    def _validate_radar_adc(self) -> None:
        if not self._radar_adc_path.is_file():
            raise FileNotFoundError(f"Radar ADC file does not exist: {self._radar_adc_path}")
        expected_size = self.radar_capture.expected_size_bytes
        if expected_size is None:  # pragma: no cover - timing validation requires a frame count.
            raise ValueError("Synchronized capture requires a finite radar frame count.")
        actual_size = self._radar_adc_path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                "Radar ADC file size does not match the declared capture contract: "
                f"{actual_size} != {expected_size} bytes."
            )

    def abort(self) -> None:
        """Close the event log while retaining an inspectable incomplete directory."""

        if self._state in {
            SoftwareCaptureState.CLOSED,
            SoftwareCaptureState.ABORTED,
        }:
            return
        self._writer.close()
        self._state = SoftwareCaptureState.ABORTED

    def _resolve_camera_frame(self, relative_path: str) -> Path:
        path = Path(relative_path)
        if not relative_path or path.is_absolute() or ".." in path.parts:
            raise ValueError("Camera frame path must be relative to the camera directory.")
        resolved = (self.camera_frames_root / path).resolve()
        try:
            resolved.relative_to(self.camera_frames_root.resolve())
        except ValueError as exc:
            raise ValueError("Camera frame path escapes the camera directory.") from exc
        return resolved

    def _resolve_capture_reference(self, relative_path: str) -> Path:
        path = Path(relative_path)
        if not relative_path or path.is_absolute() or ".." in path.parts:
            raise ValueError("Radar ADC path must be relative to the capture directory.")
        resolved = (self.capture_root / path).resolve()
        try:
            resolved.relative_to(self.capture_root.resolve())
        except ValueError as exc:
            raise ValueError("Radar ADC path escapes the capture directory.") from exc
        return resolved

    def _require_state(self, expected: SoftwareCaptureState) -> None:
        if self._state is not expected:
            raise RuntimeError(
                f"Capture {self.capture_id!r} is {self._state.value}, expected {expected.value}."
            )

    def _require_states(self, *expected: SoftwareCaptureState) -> None:
        if self._state not in expected:
            names = " or ".join(state.value for state in expected)
            raise RuntimeError(
                f"Capture {self.capture_id!r} is {self._state.value}, expected {names}."
            )


__all__ = [
    "SoftwareCaptureState",
    "SoftwareSynchronizedCapture",
]
