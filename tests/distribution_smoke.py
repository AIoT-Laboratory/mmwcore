"""Run with the installed wheel's Python -I, from outside the source checkout."""

import os
import sys
import tarfile
import tempfile
from importlib.metadata import distribution
from pathlib import Path

import numpy as np

import mmwcore
from mmwcore import _native
from mmwcore.config import RadarCaptureSpec, RadarProfile
from mmwcore.core import ADCFrameSpec, Box3D, PointCloudFrame, TrackStatus
from mmwcore.io import CompressedADCReader, compress_adc_file, decompress_adc_file
from mmwcore.tracking import ScatterBodyTracker, TiGTrack3D, TiGTrack3DSpec, TiGTrackScenery


def check_distribution(artifacts: Path) -> None:
    # Editable/source-tree imports can make a broken wheel appear usable.
    assert sys.prefix != sys.base_prefix
    for module in (mmwcore, _native):
        assert module.__file__ is not None
        assert Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())

    installed = distribution("mmwcore")
    licenses = ("LICENSE", "NOTICE", "crates/mmwcore-ti-gtrack/TI-LICENSE.txt")
    assert installed.metadata["License-Expression"] == "Apache-2.0 AND LicenseRef-TI-GTRACK"
    assert set(installed.metadata.get_all("License-File") or []) == set(licenses)
    files = installed.files
    assert files is not None
    metadata = next(
        p for p in files if p.name == "METADATA" and p.parent.name.endswith(".dist-info")
    )
    metadata_dir = Path(str(installed.locate_file(metadata))).parent
    root = Path(__file__).resolve().parents[1]
    (sdist,) = artifacts.glob("*.tar.gz")
    with tarfile.open(sdist) as source:
        prefix = f"mmwcore-{installed.version}/"
        # The Rust oracle test uses include_str! across the crate boundary.
        required = (*licenses, "tests/fixtures/ti_gtrack_3da_oracle.json", "Cargo.lock")
        for name in required:
            member = source.extractfile(prefix + name)
            assert member is not None, name
            assert member.read() == (root / name).read_bytes(), name
        for name in licenses:
            assert (metadata_dir / "licenses" / name).read_bytes() == (root / name).read_bytes()
    for name in ("py.typed", "_native.pyi"):
        assert (Path(mmwcore.__file__).parent / name).is_file()


def check_runtime() -> None:
    values = np.arange(32, dtype=np.float32).astype(np.complex64).reshape(2, 16)
    actual = _native.fft_complex_axis(values, 1, 16, 0, 0)
    np.testing.assert_allclose(actual, np.fft.fft(values, axis=1), rtol=1e-6, atol=1e-5)

    profile = RadarProfile(num_tx=1, num_rx=1, num_adc_samples=8, num_chirps_per_tx=2)
    adc = ADCFrameSpec(num_chirps=2, num_rx=1, num_samples=8)
    capture = RadarCaptureSpec(profile, adc, (0,), frame_periodicity_s=0.1, num_frames=6)
    words = np.arange(adc.raw_values_per_frame * 6, dtype="<i2")
    with tempfile.TemporaryDirectory() as directory:
        raw, archive = Path(directory) / "adc.bin", Path(directory) / "adc.mmwa"
        raw.write_bytes(words.tobytes())
        compress_adc_file(raw, archive, capture)
        frames = list(CompressedADCReader(archive).iter_frames())
        np.testing.assert_array_equal(np.concatenate([f.samples for f in frames]), words)
        assert [f.frame_id for f in frames] == list(range(6))
        restored = Path(directory) / "restored.bin"
        assert decompress_adc_file(archive, restored) == capture
        assert restored.read_bytes() == raw.read_bytes()

    # Normal wheels must use Rust even if an obsolete plugin variable is present.
    os.environ["MMWCORE_TI_GTRACK_MANIFEST"] = "missing-ti-plugin.json"
    spec = TiGTrack3DSpec(
        0.1,
        4.0,
        0.125,
        max_tracks=1,
        scenery=TiGTrackScenery(boundary_boxes=(Box3D(-10, 10, -10, 10, -10, 10),)),
    )
    with TiGTrack3D(spec) as tracker:
        assert tracker.provenance["implementation"] == "mmwcore-rust-gtrack-3da-v1"
        for frame in range(8):
            points = np.array(
                [[2 + 0.03 * frame, y, 0.1, 0.3, 1000] for y in np.linspace(-0.05, 0.05, 24)],
                dtype=np.float32,
            )
            tracks = tracker.step(
                PointCloudFrame(
                    points,
                    channels=("x", "y", "z", "velocity", "snr"),
                    coordinate_frame="sensor_forward_lateral_up",
                )
            )
        assert tracks.statuses == (TrackStatus.CONFIRMED,)
        assert np.isfinite(tracks.positions).all()

    multiscale = ScatterBodyTracker(max_bodies=1)
    points[:, 4] = 10 * np.log10(points[:, 4])  # Multiscale consumes dB, TI above uses linear SNR.
    for _ in range(3):
        *_, bodies = multiscale.step_points(points)
    assert len(bodies) == 1


if __name__ == "__main__":
    check_distribution(Path(sys.argv[1]))
    check_runtime()
    print("Installed wheel: licenses, FFT, ADC compression, TI and multiscale tracking passed.")
