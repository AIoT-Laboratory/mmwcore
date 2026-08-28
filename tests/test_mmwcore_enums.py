from __future__ import annotations

import json

import pytest

from mmwcore.core import ADCComplexLayout, TrackStatus


@pytest.mark.parametrize(
    "member",
    [
        ADCComplexLayout.GROUP2_I_THEN_Q,
        TrackStatus.CONFIRMED,
    ],
)
def test_string_enums_preserve_standard_string_semantics(
    member: ADCComplexLayout | TrackStatus,
) -> None:
    assert isinstance(member, str)
    assert str(member) == member.value
    assert f"{member}" == member.value
    assert json.dumps(member) == json.dumps(member.value)
