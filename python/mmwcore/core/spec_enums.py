"""Common enum values for mmwcore processing specs."""

from __future__ import annotations

from mmwcore._compat import StrEnum


class ADCComplexLayout(StrEnum):
    """Raw int16 layout used to reconstruct complex ADC samples."""

    IQ_INTERLEAVED = "iq_interleaved"  # I1 Q1 I2 Q2 ...
    SAMPLE_I_THEN_Q = "sample_i_then_q"  # RX0I RX1I ... RX0Q RX1Q ...
    GROUP2_I_THEN_Q = "group2_i_then_q"  # I1 I2 Q1 Q2 ...


class FFTWindow(StrEnum):
    """Window functions available for FFT preprocessing."""

    NONE = "none"
    HANN = "hann"
    HAMMING = "hamming"


class DetectionMethod(StrEnum):
    """Detector family used by the offline processing pipeline."""

    THRESHOLD = "threshold"
    CFAR = "cfar"


class CFARMode(StrEnum):
    """Noise-window reduction modes for cell-averaging CFAR variants."""

    CA = "ca"
    GO = "go"
    SO = "so"
    CACC = "cacc"


class CFARInputScale(StrEnum):
    """Signal scale supplied to a CFAR detector."""

    MAGNITUDE = "magnitude"
    POWER = "power"
