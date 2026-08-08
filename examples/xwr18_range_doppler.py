"""Build an explicit standard XWR1843 EVM recipe for an xWR18xx capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmwcore import open_capture
from mmwcore.config import xwr1843_evm_antenna_geometry
from mmwcore.core import ADCDecodeRecipe, DopplerFFTSpec, RangeDopplerRecipe, TDMVirtualArraySpec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()
    capture = open_capture(args.capture)
    if capture.raw_capture.family != "xwr18xx":
        raise SystemExit(f"expected xwr18xx, got {capture.raw_capture.family!r}")
    geometry = xwr1843_evm_antenna_geometry()
    tdm = TDMVirtualArraySpec(geometry, capture.radar_capture.tx_order)
    recipe = RangeDopplerRecipe(
        decode=ADCDecodeRecipe(capture.radar_capture.adc),
        doppler_fft=DopplerFFTSpec(input_axis="loop"),
        tdm_virtual_array=tdm,
    )
    cube = capture.range_doppler(recipe, frame_index=args.frame)
    print(
        json.dumps(
            {
                "family": capture.raw_capture.family,
                "geometry": geometry.name,
                "tx_order": list(tdm.tx_order),
                "virtual_antennas": tdm.num_virtual_antennas,
                "axes": list(cube.axes),
                "shape": list(cube.data.shape),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
