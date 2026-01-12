from __future__ import annotations

from annolid.annotation.sanitize_labelme_json import sanitize_labelme_payload


def test_sanitize_labelme_payload_moves_non_bool_flags() -> None:
    payload = {
        "flags": {"ok": "yes", "confidence": 0.2},
        "shapes": [
            {
                "label": "nose",
                "shape_type": "point",
                "points": [[1.0, 2.0]],
                "flags": {"instance_label": "animal", "keep": True, "score": 0.9},
            }
        ],
    }
    sanitized, changed = sanitize_labelme_payload(payload)
    assert changed is True

    assert sanitized["flags"] == {"ok": True}
    assert sanitized["annolid_flags_meta"] == {"confidence": 0.2}

    shape0 = sanitized["shapes"][0]
    assert shape0["flags"] == {"keep": True}
    assert shape0["instance_label"] == "animal"
    assert shape0["score"] == 0.9

