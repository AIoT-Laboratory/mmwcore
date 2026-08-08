"""Shared Range-Doppler binding for validated mmwcli capture contracts."""

from __future__ import annotations

from typing import Protocol

from mmwcore.config import RadarCaptureSpec, RadarProfile
from mmwcore.core import ADCComplexLayout, RangeDopplerRecipe


class RangeDopplerPreset(Protocol):
    """Build a recipe from the physical fields proven by a capture contract."""

    def __call__(
        self,
        profile: RadarProfile,
        *,
        adc_layout: ADCComplexLayout,
        tx_order: tuple[int, ...],
    ) -> RangeDopplerRecipe: ...


def _resolve_range_doppler_recipe(
    contract: RadarCaptureSpec,
    binding: RangeDopplerRecipe | RangeDopplerPreset | None,
    *,
    context: str,
) -> RangeDopplerRecipe | None:
    if binding is None:
        return None
    if isinstance(binding, RangeDopplerRecipe):
        recipe = binding
    elif callable(binding):
        recipe = binding(
            contract.profile,
            adc_layout=contract.adc.layout,
            tx_order=contract.tx_order,
        )
    else:
        raise TypeError(
            f"{context} range_doppler must be a RangeDopplerRecipe, preset callable, or None."
        )
    _validate_range_doppler_recipe(contract, recipe, context=context)
    return recipe


def _validate_range_doppler_recipe(
    contract: RadarCaptureSpec,
    recipe: RangeDopplerRecipe,
    *,
    context: str,
) -> None:
    if not isinstance(recipe, RangeDopplerRecipe):
        raise TypeError(f"{context} range_doppler requires a RangeDopplerRecipe.")
    if recipe.decode.adc != contract.adc:
        raise ValueError("Range-Doppler recipe ADC spec does not match the capture contract.")

    tdm = recipe.tdm_virtual_array
    if tdm is None:
        if len(contract.tx_order) > 1:
            raise ValueError("Multi-Tx capture processing requires an explicit TDM virtual array.")
        if recipe.doppler_fft.input_axis != "chirp":
            raise ValueError("Single-Tx capture processing requires the chirp Doppler axis.")
        return
    if tdm.tx_order != contract.tx_order:
        raise ValueError("Range-Doppler recipe Tx order does not match the capture contract.")
    if tdm.geometry.num_rx != contract.profile.num_rx:
        raise ValueError(
            "Range-Doppler recipe receiver geometry does not match the capture contract."
        )


__all__ = ["RangeDopplerPreset"]
