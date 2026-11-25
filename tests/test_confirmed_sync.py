"""The analysis must use exactly the synchronisation the user confirmed in the app."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gazeplay.config import Settings  # noqa: E402
from gazeplay.sync import resolve_sync  # noqa: E402
from gazeplay.web.server import apply_confirmed_sync  # noqa: E402

EGO, ROOM = {"duration_s": 70.0}, {"duration_s": 80.0}


def confirmed(**kw):
    d = {"confirmed": True, "offset": 7.31, "ego_start": 16.68, "ego_end": 48.661,
         "audio_offset": 7.299, "audio_confidence": "high"}
    d.update(kw)
    return d


def test_values_used_exactly():
    s = Settings.from_preset("balanced")
    apply_confirmed_sync(s, confirmed(), EGO, ROOM)
    assert s.sync_mode == "segments" and s.sync_offset_s == 7.31
    assert (s.ego_start_s, s.ego_end_s) == (16.68, 48.661)
    assert s.room_start_s == pytest.approx(16.68 + 7.31) and s.room_end_s == pytest.approx(48.661 + 7.31)
    res = resolve_sync(s, "", "")
    assert res.offset_s == 7.31 and res.method == "segments"
    assert "+0.011" in res.message   # the manual adjustment relative to the audio sync is reported


@pytest.mark.parametrize("bad, msg", [
    (None, "confirm"),
    ({"offset": 7.3, "ego_start": 0, "ego_end": 10}, "confirm"),          # not confirmed
    (confirmed(offset="x"), "incomplete"),
    (confirmed(ego_start=-5), "outside"),
    (confirmed(ego_end=90), "outside"),
    (confirmed(offset=40, ego_start=0, ego_end=70), "room video"),        # room segment past its end
    (confirmed(ego_start=10, ego_end=10.5), "shorter"),
])
def test_rejected(bad, msg):
    with pytest.raises(ValueError, match=msg):
        apply_confirmed_sync(Settings.from_preset("balanced"), bad, EGO, ROOM)


def test_negative_offset_and_rounding_tolerance():
    s = Settings.from_preset("balanced")
    # room camera started 3 s AFTER the glasses; ego_start 2.98 is 0.02 s before the room video exists
    apply_confirmed_sync(s, confirmed(offset=-3.0, ego_start=2.98, ego_end=70.02, audio_offset=None), EGO, ROOM)
    assert s.ego_start_s == 3.0 and s.ego_end_s == 70.0 and s.room_start_s == 0.0
    assert s.audio_offset_s is None


def test_new_defaults():
    s = Settings.from_preset("balanced")
    assert s.include_people is False and s.child_selection == "only"
