from __future__ import annotations

import os
import subprocess
import sys
import threading

import pytest

from annolid.yolo import ultralytics_cli as cli


@pytest.fixture
def processes(monkeypatch):
    launched = []
    original = subprocess.Popen

    def launch(*args, **kwargs):
        proc = original(*args, **kwargs)
        launched.append(proc)
        return proc

    monkeypatch.setattr(cli.subprocess, "Popen", launch)
    yield launched
    for proc in launched:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)


def test_streams_output_and_retains_bounded_tail():
    lines = []
    result = cli.run_yolo_cli(
        [sys.executable, "-c", "print('first'); print('second'); print('third')"],
        output_sink=lines.append,
        tail_lines=2,
    )
    assert result.returncode == 0
    assert lines == ["first\n", "second\n", "third\n"]
    assert result.output_tail == ["second", "third"]


@pytest.mark.parametrize("close_output", [False, True])
def test_cancels_quiet_process_and_reaps_it(processes, close_output):
    ready = threading.Event()
    stop = threading.Event()
    errors = []
    script = "import os, time; print('ready', flush=True); "
    if close_output:
        script += "os.close(1); os.close(2); "
    script += "time.sleep(60)"

    def run():
        try:
            cli.run_yolo_cli(
                [sys.executable, "-c", script],
                stop_event=stop,
                output_sink=lambda line: ready.set(),
            )
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        assert ready.wait(timeout=5), "Child did not start"
        stop.set()
        worker.join(timeout=3)
        assert not worker.is_alive(), "Cancellation blocked on subprocess output"
        assert len(errors) == 1
        assert str(errors[0]) == "YOLO training cancelled."
        assert processes[0].poll() is not None
    finally:
        if processes and processes[0].poll() is None:
            processes[0].kill()
        worker.join(timeout=5)


def test_output_sink_failure_terminates_process(processes):
    def sink(line):
        raise ValueError("Cannot write training log")

    with pytest.raises(ValueError, match="Cannot write training log"):
        cli.run_yolo_cli(
            [
                sys.executable,
                "-c",
                "import time; print('ready', flush=True); time.sleep(60)",
            ],
            output_sink=sink,
        )
    assert processes[0].poll() is not None


def test_pre_cancelled_job_does_not_launch(processes):
    stop = threading.Event()
    stop.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        cli.run_yolo_cli([sys.executable, "-c", "pass"], stop_event=stop)
    assert processes == []


def test_nonzero_exit_keeps_diagnostic_output():
    result = cli.run_yolo_cli(
        [
            sys.executable,
            "-c",
            "import sys; print('failed', file=sys.stderr); sys.exit(7)",
        ],
    )
    assert result.returncode == 7
    assert result.output_tail == ["failed"]


def test_invalid_output_bytes_do_not_abort_logging():
    result = cli.run_yolo_cli(
        [sys.executable, "-c", "import os; os.write(1, b'bad: \\xff\\n')"],
        env={"PYTHONIOENCODING": "utf-8"},
    )
    assert result.returncode == 0
    assert len(result.output_tail) == 1


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal handling")
def test_sink_failure_kills_process_that_ignores_sigterm(processes):
    script = (
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('ready', flush=True); time.sleep(60)"
    )

    def sink(line):
        raise ValueError("sink failed")

    with pytest.raises(ValueError, match="sink failed"):
        cli.run_yolo_cli([sys.executable, "-c", script], output_sink=sink)
    assert processes[0].poll() is not None


def test_invalid_tail_size_does_not_launch(processes):
    with pytest.raises(ValueError, match="tail_lines"):
        cli.run_yolo_cli([sys.executable, "-c", "pass"], tail_lines=0)
    assert processes == []
