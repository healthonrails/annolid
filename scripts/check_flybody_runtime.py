#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Any, Callable


def _resolve_dotted_callable(path: str) -> Callable[..., Any]:
    module_name, sep, attr_name = str(path or "").partition(":")
    if not module_name or not sep or not attr_name:
        raise ValueError(
            "Expected dotted callable in the form 'module.submodule:attribute'"
        )
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name, None)
    if not callable(value):
        raise AttributeError(f"Callable not found: {path}")
    return value


def _axis_names(axis: Any) -> list[str]:
    names = getattr(axis, "_names_to_offsets", None)
    if isinstance(names, dict):
        return list(names.keys())
    names = getattr(axis, "_names_to_indices", None)
    if isinstance(names, dict):
        return list(names.keys())
    return []


def _python_support_note() -> str | None:
    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 13):
        return (
            "Python 3.13 can require a local labmaze compatibility workaround. "
            "Prefer Python 3.10 to 3.12 for a clean FlyBody environment."
        )
    return None


def probe_runtime(
    *,
    env_factory_path: str,
    ik_function_path: str,
) -> dict[str, Any]:
    env_factory = _resolve_dotted_callable(env_factory_path)
    ik_solver = _resolve_dotted_callable(ik_function_path)
    env = env_factory()
    physics = getattr(env, "physics", env)
    walker = getattr(getattr(env, "task", None), "_walker", None)
    mjcf_model = getattr(walker, "mjcf_model", None)
    walker_joint_names = []
    if mjcf_model is not None and hasattr(mjcf_model, "find_all"):
        walker_joint_names = [
            str(getattr(joint, "name", "") or "").strip()
            for joint in mjcf_model.find_all("joint")
            if str(getattr(joint, "name", "") or "").strip()
        ]
    site_names = _axis_names(physics.named.model.site_pos.axes.row)
    dof_joint_names = _axis_names(physics.named.model.dof_jntid.axes.row)
    close = getattr(env, "close", None)
    if callable(close):
        close()
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "env_factory": env_factory_path,
        "ik_function": ik_function_path,
        "environment_type": type(env).__name__,
        "task_type": type(getattr(env, "task", None)).__name__,
        "ik_callable_module": getattr(ik_solver, "__module__", ""),
        "walker_joint_count": len(walker_joint_names),
        "walker_joint_sample": walker_joint_names[:8],
        "runtime_site_count": len(site_names),
        "runtime_site_sample": site_names[:8],
        "runtime_dof_joint_count": len(dof_joint_names),
        "runtime_dof_joint_sample": dof_joint_names[:8],
        "python_support_note": _python_support_note(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that the optional FlyBody runtime can import and build an environment."
    )
    parser.add_argument(
        "--env-factory",
        default="flybody.fly_envs:walk_imitation",
        help="Dotted callable used to construct the FlyBody environment.",
    )
    parser.add_argument(
        "--ik-function",
        default="flybody.inverse_kinematics:qpos_from_site_xpos",
        help="Dotted callable used for FlyBody IK.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of text.",
    )
    args = parser.parse_args()

    try:
        result = probe_runtime(
            env_factory_path=args.env_factory,
            ik_function_path=args.ik_function,
        )
    except Exception as exc:
        message = {
            "ok": False,
            "error": str(exc),
            "python_support_note": _python_support_note(),
        }
        if args.json:
            print(json.dumps(message, indent=2, sort_keys=True))
        else:
            print("FlyBody runtime check failed.")
            print(f"Error: {exc}")
            if message["python_support_note"]:
                print(message["python_support_note"])
        return 1

    if args.json:
        payload = {"ok": True, **result}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print("FlyBody runtime check passed.")
    print(f"Python: {result['python']}")
    print(f"Environment factory: {result['env_factory']}")
    print(f"IK function: {result['ik_function']}")
    print(f"Environment: {result['environment_type']} / task: {result['task_type']}")
    print(
        "Walker joints: "
        f"{result['walker_joint_count']} total; sample={result['walker_joint_sample']}"
    )
    print(
        "Runtime sites: "
        f"{result['runtime_site_count']} total; sample={result['runtime_site_sample']}"
    )
    print(
        "Runtime DOF joints: "
        f"{result['runtime_dof_joint_count']} total; sample={result['runtime_dof_joint_sample']}"
    )
    if result["python_support_note"]:
        print(result["python_support_note"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
