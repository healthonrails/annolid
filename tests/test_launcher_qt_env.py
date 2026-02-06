import os

from annolid.gui.launcher import _sanitize_qt_plugin_env


def test_sanitize_qt_env_removes_cv2_platform_paths_on_linux() -> None:
    env = {
        "QT_QPA_PLATFORM_PLUGIN_PATH": "/tmp/venv/lib/python3.11/site-packages/cv2/qt/plugins",
        "QT_QPA_FONTDIR": "/tmp/venv/lib/python3.11/site-packages/cv2/qt/fonts",
    }

    _sanitize_qt_plugin_env(env, is_linux=True)

    assert "QT_QPA_PLATFORM_PLUGIN_PATH" not in env
    assert "QT_QPA_FONTDIR" in env


def test_sanitize_qt_env_prunes_only_cv2_entries_from_qt_plugin_path() -> None:
    env = {
        "QT_PLUGIN_PATH": os.pathsep.join(
            [
                "/opt/qt/plugins",
                "/tmp/venv/lib/python3.11/site-packages/cv2/qt/plugins",
                "/usr/lib/qt/plugins",
            ]
        )
    }

    _sanitize_qt_plugin_env(env, is_linux=True)

    assert env["QT_PLUGIN_PATH"] == os.pathsep.join(
        ["/opt/qt/plugins", "/usr/lib/qt/plugins"]
    )


def test_sanitize_qt_env_is_noop_when_not_linux() -> None:
    original = "/tmp/venv/lib/python3.11/site-packages/cv2/qt/plugins"
    env = {"QT_QPA_PLATFORM_PLUGIN_PATH": original}

    _sanitize_qt_plugin_env(env, is_linux=False)

    assert env["QT_QPA_PLATFORM_PLUGIN_PATH"] == original
