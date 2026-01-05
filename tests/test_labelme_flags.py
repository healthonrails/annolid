from annolid.utils.labelme_flags import sanitize_labelme_flags


def test_sanitize_labelme_flags_drops_non_boolean_values() -> None:
    flags = {
        "rearing": True,
        "sniffing": "true",
        "exploring": "0",
        "instance_label": "mouse",
        "": True,
        None: True,
    }
    assert sanitize_labelme_flags(flags) == {
        "rearing": True,
        "sniffing": True,
        "exploring": False,
    }
