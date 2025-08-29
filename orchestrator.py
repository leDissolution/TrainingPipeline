import argparse
import glob
import os
import shlex
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import itertools
from pathlib import Path

import yaml
import subprocess
import re


@dataclass
class Stage:
    name: str
    kind: str  # 'script' | 'make_sweeps'
    script: Optional[str]
    venv: Optional[str]
    args: Dict[str, Any]
    env: Dict[str, str]
    sweeps: Optional[Dict[str, Any]]


def _is_windows() -> bool:
    return os.name == "nt"


def _venv_python(venv_path: str) -> str:
    # Return Python executable path for a venv
    if _is_windows():
        cand = os.path.join(venv_path, "Scripts", "python.exe")
    else:
        cand = os.path.join(venv_path, "bin", "python")
    return cand


def _normpath(base_dir: str, rel: str) -> str:
    if os.path.isabs(rel):
        return os.path.normpath(rel)
    return os.path.normpath(os.path.join(base_dir, rel))


def _load_pipeline(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _interpolate_string(s: str, vars_map: Dict[str, str]) -> str:
    """Replace ${VAR} occurrences in s using vars_map (leave unknowns intact).
    Performs up to 3 passes to allow simple nested expansions.
    """
    pattern = re.compile(r"\$\{([^}]+)\}")
    result = s
    for _ in range(3):
        def repl(m: re.Match[str]) -> str:
            key = m.group(1)
            return str(vars_map.get(key, m.group(0)))
        new_result = pattern.sub(repl, result)
        if new_result == result:
            break
        result = new_result
    return result


def _deep_interpolate(obj: Any, vars_map: Dict[str, str]) -> Any:
    """Recursively interpolate strings in dicts/lists using ${VAR} syntax."""
    if isinstance(obj, str):
        return _interpolate_string(obj, vars_map)
    if isinstance(obj, list):
        return [_deep_interpolate(x, vars_map) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_deep_interpolate(list(obj), vars_map))
    if isinstance(obj, dict):
        return {k: _deep_interpolate(v, vars_map) for k, v in obj.items()}
    return obj


def _to_str(v: Any) -> str:
    if isinstance(v, bool):
        # Use explicit true/false to be parser-agnostic
        return "true" if v else "false"
    return str(v)


def _flatten_args(arg_map: Dict[str, Any]) -> List[str]:
    """Convert {"--flag": value, "--list": [a,b]} into a flat argv list.
    Rules:
      - bool -> pass as "true"/"false" value pair
      - list/tuple -> repeat flag for each item
      - None -> include just the flag (toggle)
      - other scalars -> flag value
    """
    argv: List[str] = []
    for k, v in arg_map.items():
        if v is None:
            argv.append(k)
        elif isinstance(v, (list, tuple)):
            for item in v:
                argv.extend([k, _to_str(item)])
        else:
            argv.extend([k, _to_str(v)])
    return argv


def _list_config_files(dir_path: str, patterns: Optional[List[str]] = None) -> List[str]:
    patterns = patterns or ["*.yaml", "*.yml"]
    out: List[str] = []
    for pat in patterns:
        out.extend(sorted(glob.glob(os.path.join(dir_path, pat))))
    # de-dupe while preserving order
    seen = set()
    unique = []
    for p in out:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def _expand_sweeps(base_dir: str, args_map: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand any value like {'sweep': '<dir>', 'glob': '*.yaml'} into multiple runs.
    If 'glob' is provided, collect matches using that pattern(s).
    If 'glob' is omitted, sweep over any first-level children (files or directories)
    inside the given directory. Cross-product if multiple sweeps present.
    """
    runs: List[Dict[str, Any]] = [deepcopy(args_map)]
    for flag, val in list(args_map.items()):
        if isinstance(val, dict) and ("sweep" in val or "glob" in val):
            sweep_dir = _normpath(base_dir, val.get("sweep", "."))
            globs = val.get("glob", None)
            files: List[str]
            if globs is not None:
                if isinstance(globs, str):
                    patterns = [globs]
                elif isinstance(globs, (list, tuple)):
                    patterns = list(globs)
                else:
                    patterns = ["*"]
                files = _list_config_files(sweep_dir, patterns)
                if not files:
                    print(f"[orchestrator] Warning: sweep for {flag} found no files in {sweep_dir} with {patterns}")
            else:
                # No glob specified: enumerate any first-level children (files or directories)
                try:
                    entries = sorted(os.listdir(sweep_dir))
                except FileNotFoundError:
                    entries = []
                files = [os.path.join(sweep_dir, e) for e in entries]
                if not files:
                    print(f"[orchestrator] Warning: sweep for {flag} found no entries in {sweep_dir}")
            new_runs: List[Dict[str, Any]] = []
            for r in runs:
                for f in files:
                    nr = deepcopy(r)
                    nr[flag] = f
                    new_runs.append(nr)
            runs = new_runs
    return runs


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _run_subprocess(cmd: List[str], env: Dict[str, str], cwd: str, log_path: Optional[str], dry_run: bool) -> int:
    if dry_run:
        printable = " ".join(shlex.quote(c) for c in cmd)
        env_subset = {k: v for k, v in env.items() if k.startswith("CUDA_") or k.startswith("PIPELINE_") or k.startswith("STAGE_") or k.startswith("RUN_")}
        print(f"[DRY-RUN] cwd={cwd}\n         env+={env_subset}\n         cmd={printable}")
        return 0

    # Stream output to console + optional log
    log_file = open(log_path, "a", encoding="utf-8") if log_path else None
    try:
        print(f"[orchestrator] Running: {' '.join(cmd)} (cwd={cwd})")
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip("\n")
            print(line)
            if log_file:
                log_file.write(line + "\n")
        process.wait()
        return process.returncode or 0
    finally:
        if log_file:
            log_file.flush()
            log_file.close()


def _collect_stages(pipeline_yaml: str, data: Dict[str, Any]) -> Tuple[str, List[Stage]]:
    pipeline_dir = os.path.dirname(os.path.abspath(pipeline_yaml))
    repo_root = os.path.abspath(os.path.join(pipeline_dir, ".."))
    stages: List[Stage] = []
    for i, st in enumerate(data.get("stages", []) or []):
        name = str(st.get("name", f"stage_{i}"))

        # Normalize env vars early
        env_vars: Dict[str, str] = {}
        for e in st.get("env_variables", []) or []:
            if isinstance(e, dict) and "name" in e:
                env_vars[str(e["name"])] = _to_str(e.get("value", ""))

        if "make_sweeps" in st and st.get("make_sweeps") is not None:
            sweeps_cfg = st.get("make_sweeps") or {}
            stages.append(
                Stage(
                    name=name,
                    kind="make_sweeps",
                    script=None,
                    venv=None,
                    args={},
                    env=env_vars,
                    sweeps=sweeps_cfg,
                )
            )
            continue

        script_info = st.get("script", {}) or {}
        script = str(script_info.get("name"))
        venv = script_info.get("venv")
        args_map = script_info.get("args", {}) or {}

        # Note: resolve script relative to repo_root (so 'train.py' works),
        # but allow paths like '../DatasetProcessor/prepare_dataset.py'.
        abs_script = _normpath(repo_root, script)
        stages.append(
            Stage(
                name=name,
                kind="script",
                script=abs_script,
                venv=str(venv) if venv else None,
                args=args_map,
                env=env_vars,
                sweeps=None,
            )
        )
    return repo_root, stages


def _set_dotted(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    """Set a nested value into cfg following a dotted path that may include
    bracket selectors, e.g. 'a.b["x"].c[0].d'.

    Special handling:
      - When encountering 'groups["<name>"]', treat 'groups' as a list of
        dicts and select/create the element whose 'name' equals <name>.

    - Dot segments separate path components
    - Brackets can contain quoted strings or integers for list indices
    - Intermediate structures are created as needed (dict or list)
    """

    def _parse_path(path: str) -> List[Any]:
        parts: List[Any] = []
        # split first on dots, then further split each token's bracket selectors
        dot_segs = path.split(".") if path else []
        for seg in dot_segs:
            if not seg:
                continue
            # leading name up to first '['
            j = 0
            while j < len(seg) and seg[j] != '[':
                j += 1
            if j > 0:
                name = seg[:j]
                if name:
                    parts.append(name)
            # parse zero or more [ ... ] selectors
            while j < len(seg):
                if seg[j] != '[':
                    break
                j += 1  # skip '['
                # find matching ']'
                k = j
                # we assume no nested brackets inside a selector
                while k < len(seg) and seg[k] != ']':
                    k += 1
                inner = seg[j:k].strip()
                # move past ']'
                j = k + 1 if k < len(seg) else k
                # interpret inner as quoted string or int
                if (len(inner) >= 2 and ((inner[0] == inner[-1] == '"') or (inner[0] == inner[-1] == "'"))):
                    parts.append(inner[1:-1])
                else:
                    try:
                        parts.append(int(inner))
                    except Exception:
                        # treat as raw string key (e.g., unquoted)
                        if inner:
                            parts.append(inner)
            # any trailing characters after brackets are treated as plain text
            if j < len(seg):
                tail = seg[j:]
                if tail:
                    parts.append(tail)
        return parts

    keys = _parse_path(dotted)
    if not keys:
        return
    cur: Any = cfg
    i = 0
    # Walk through all but last key
    while i < len(keys) - 1:
        k = keys[i]
        nxt = keys[i + 1]

        if isinstance(k, int):
            # current level must be a list
            if not isinstance(cur, list):
                return
            while len(cur) <= k:
                cur.append({} if not isinstance(nxt, int) else [])
            cur = cur[k]
            i += 1
            continue

        # k is str -> dict key
        if not isinstance(cur, dict):
            return

        # Special case: selecting by name in 'groups["<name>"]'
        if k == "groups" and isinstance(nxt, str):
            # ensure list at cur['groups']
            if "groups" not in cur or not isinstance(cur["groups"], list):
                cur["groups"] = []
            lst = cur["groups"]
            # find element by name
            sel_name = nxt
            found = None
            for elem in lst:
                if isinstance(elem, dict) and elem.get("name") == sel_name:
                    found = elem
                    break
            if found is None:
                found = {"name": sel_name}
                lst.append(found)
            cur = found
            # we consumed both 'groups' and the selector name
            i += 2
            continue

        # Default path building for dicts/lists
        if k not in cur or not isinstance(cur[k], (dict, list)):
            cur[k] = ([] if isinstance(nxt, int) else {})
        cur = cur[k]
        i += 1

    # Set the final key
    last = keys[-1]
    if isinstance(last, int):
        if not isinstance(cur, list):
            return
        while len(cur) <= last:
            cur.append(None)
        cur[last] = value
    else:
        if not isinstance(cur, dict):
            return
        cur[last] = value


def _format_lr_token(v: float) -> str:
    # Try to match pattern like 3e-5 -> lr3e5
    if v == 0:
        return "lr0"
    import math

    exp = int(math.floor(math.log10(abs(v))))
    mant = v / (10 ** exp)
    # Snap exponent -5 style if close
    if exp == -5 or abs(exp + 5) <= 1:
        # round mant to nearest int if close
        m_rounded = int(round(mant))
        if abs(mant - m_rounded) < 1e-8 and m_rounded > 0:
            return f"lr{m_rounded}e5"
    # Fallback compact
    s = f"{v:.0e}" if v >= 1e-4 else f"{v:.0e}"
    s = s.replace("-0", "").replace("e-0", "e")
    s = s.replace("e-05", "e5").replace("e-5", "e5")
    s = s.replace("e+00", "")
    return f"lr{s}"


def _value_token(key: str, val: Any) -> str:
    k = key
    if isinstance(val, float):
        if k.endswith("layerwise_lr.base_lr"):
            return _format_lr_token(float(val))
        if k.endswith("tokens.grad_boost.factor"):
            return f"gb{int(round(val))}"
        # generic float
        return str(val).replace(".", "p").replace("-", "m")
    if isinstance(val, (int, bool)):
        prefix = "r" if k.endswith("peft.r") else "v"
        return f"{prefix}{val}"
    # strings and others
    return str(val).replace("/", "_").replace(" ", "_")


def _make_swipe_name(pairs: List[Tuple[str, Any]]) -> str:
    def _extract_group_name(k: str) -> Optional[str]:
        # Look for groups["name"] or groups['name'] or groups[name]
        import re
        m = re.search(r"groups\[(?:\"([^\"]+)\"|'([^']+)'|([^\]]+))\]", k)
        if not m:
            return None
        return m.group(1) or m.group(2) or (m.group(3).strip() if m.group(3) else None)

    parts: List[str] = []
    for k, v in pairs:
        tok = _value_token(k, v)
        # If token already has a semantic prefix (lr, gb, r), don't add key
        grp = _extract_group_name(k)
        if tok.startswith(("lr", "gb", "r")):
            parts.append(tok if not grp else f"{grp}_{tok}")
        else:
            suffix = k.split(".")[-1]
            base = f"{suffix}-{tok}"
            parts.append(f"{grp}_{base}" if grp else base)
    return "_".join(parts)


def _generate_sweeps(repo_root: str, sweeps_cfg: Dict[str, Any], vars_map: Optional[Dict[str, str]] = None) -> List[str]:
    base_config_rel = str(sweeps_cfg.get("base_config", "")).strip()
    if vars_map:
        base_config_rel = _interpolate_string(base_config_rel, vars_map)
    if not base_config_rel:
        raise ValueError("make_sweeps.base_config is required")
    base_config_abs = _normpath(repo_root, base_config_rel)
    base_dir = os.path.dirname(base_config_abs)

    out_name = str(sweeps_cfg.get("name", "sweep")).strip() or "sweep"
    out_dir = os.path.join(base_dir, out_name)
    _ensure_dir(out_dir)

    # Clear old YAMLs in the sweep directory
    removed = 0
    for fname in os.listdir(out_dir):
        if fname.lower().endswith(('.yaml', '.yml')):
            try:
                os.remove(os.path.join(out_dir, fname))
                removed += 1
            except Exception as e:
                print(f"[orchestrator] Warning: could not remove {fname}: {e}")
    if removed:
        print(f"[orchestrator] make_sweeps: cleared {removed} existing YAMLs in {out_dir}")

    # Allow interpolation inside param_ranges values that are strings
    param_ranges = sweeps_cfg.get("param_ranges", {}) or {}
    if not isinstance(param_ranges, dict) or not param_ranges:
        print("[orchestrator] make_sweeps: no param_ranges provided, nothing to do.")
        return []

    # Prepare cross product
    keys = list(param_ranges.keys())
    values_lists: List[List[Any]] = []
    for v in param_ranges.values():
        seq = list(v)
        if vars_map:
            seq = [_deep_interpolate(x, vars_map) if isinstance(x, (str, list, dict)) else x for x in seq]
        values_lists.append(seq)
    created: List[str] = []

    for combo in itertools.product(*values_lists):
        pairs = list(zip(keys, combo))

        # Skip invalid layer ranges where corresponding '.layers.start' and '.layers.end'
        # exist for the same path and end <= start.
        # Build a quick lookup and detect matching prefixes ending in '.layers'.
        kv = {k: v for k, v in pairs}
        # collect potential prefixes
        prefixes = set()
        for k in kv.keys():
            if k.endswith('.layers.start'):
                prefixes.add(k.rsplit('.start', 1)[0])  # yields '...layers'
            elif k.endswith('.layers.end'):
                prefixes.add(k.rsplit('.end', 1)[0])
        invalid = False
        for pref in prefixes:
            ks = f"{pref}.start"
            ke = f"{pref}.end"
            if ks in kv and ke in kv:
                try:
                    s_val = int(kv[ks])
                    e_val = int(kv[ke])
                    if e_val <= s_val:
                        invalid = True
                        break
                except Exception:
                    # if values aren't integers, don't enforce
                    pass
        if invalid:
            continue
        overrides: Dict[str, Any] = {}
        for k, v in pairs:
            _set_dotted(overrides, k, v)

        swipe = _make_swipe_name(pairs)

        # Build YAML structure
        # extends should be relative from out_dir to base_config_abs
        rel_extends = os.path.relpath(base_config_abs, out_dir)
        yaml_obj: Dict[str, Any] = {
            "extends": rel_extends,
            "swipe_name": swipe,
        }
        # merge overrides at top-level
        for topk, topv in overrides.items():
            yaml_obj.setdefault(topk, topv)

        # Write file
        file_name = f"{swipe}.yaml"
        file_path = os.path.join(out_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_obj, f, sort_keys=False)
        created.append(file_path)

    print(f"[orchestrator] make_sweeps: created {len(created)} files under {out_dir}")
    return created


def run_pipeline(pipeline_yaml: str, stop_on_fail: bool = True, dry_run: bool = False, log_dir: Optional[str] = None, stage_filter: Optional[Iterable[str]] = None, start_stage: Optional[str] = None) -> int:
    data = _load_pipeline(pipeline_yaml)
    # Compute locations early
    pipeline_dir = os.path.dirname(os.path.abspath(pipeline_yaml))
    repo_root = os.path.abspath(os.path.join(pipeline_dir, ".."))

    # Load optional environment config
    env_vars: Dict[str, Any] = {}
    env_name: Optional[str] = None
    env_cfg = data.get("environment")
    if isinstance(env_cfg, str) and env_cfg.strip():
        env_path = _normpath(repo_root, env_cfg.strip())
        try:
            with open(env_path, "r", encoding="utf-8") as ef:
                loaded = yaml.safe_load(ef) or {}
            if isinstance(loaded, dict):
                env_name = str(loaded.get("name") or loaded.get("env") or loaded.get("env_name") or "").strip() or None
                env_vars = dict(loaded.get("vars", {})) if isinstance(loaded.get("vars"), dict) else {
                    k: v for k, v in loaded.items() if k not in {"name", "env", "env_name"}
                }
        except Exception as e:
            print(f"[orchestrator] Warning: failed to load environment config '{env_cfg}': {e}")

    # Pipeline-level vars (optional)
    pipeline_vars = data.get("vars", {}) if isinstance(data.get("vars"), dict) else {}

    # Built-ins and scope assembly (precedence: OS env > pipeline vars > env file vars > built-ins)
    ts = time.strftime("%Y%m%d-%H%M%S")
    builtins_vars: Dict[str, str] = {
        "REPO_ROOT": repo_root,
        "PIPELINE_DIR": pipeline_dir,
        "TS": ts,
    }
    scope_vars: Dict[str, str] = {}
    # start with built-ins
    scope_vars.update({k: str(v) for k, v in builtins_vars.items()})
    # then environment config vars
    scope_vars.update({str(k): str(v) for k, v in env_vars.items()})
    # then pipeline-level vars
    scope_vars.update({str(k): str(v) for k, v in pipeline_vars.items()})
    # finally, OS env wins
    scope_vars.update({k: v for k, v in os.environ.items()})

    # Prepare logs (allow LOGS_BASE override)
    resolved_logs_base = scope_vars.get("LOGS_BASE")
    if log_dir:
        base_log_dir = log_dir
    elif resolved_logs_base:
        base_log_dir = os.path.join(_interpolate_string(resolved_logs_base, scope_vars), ts)
    else:
        base_log_dir = os.path.join(repo_root, "pipeline_runs", ts)
    _ensure_dir(base_log_dir)

    # Choose default python
    default_python = sys.executable or ("py" if _is_windows() else "python3")

    # Collect stages after we have repo_root (no var expansion yet)
    _, stages = _collect_stages(pipeline_yaml, data)

    # Determine start index if a starting stage was provided
    start_index = 0
    if start_stage:
        try:
            start_index = next(i for i, s in enumerate(stages) if s.name == start_stage)
        except StopIteration:
            print(f"[orchestrator] Error: --start-stage '{start_stage}' not found among defined stages: {[s.name for s in stages]}")
            return 1

    # Apply filters: if --stages provided, intersect with tail starting from start_index
    if stage_filter:
        selected = [s for i, s in enumerate(stages) if i >= start_index and s.name in stage_filter]
    else:
        selected = [s for i, s in enumerate(stages) if i >= start_index]
    if not selected:
        print("[orchestrator] No matching stages to run.")
        return 0

    overall_rc = 0
    for s_idx, st in enumerate(selected):
        print(f"\n[orchestrator] === Stage {s_idx+1}/{len(selected)}: {st.name} ===")

        # Handle make_sweeps stages
        if st.kind == "make_sweeps":
            # Logs directory for stage
            stage_log_dir = os.path.join(base_log_dir, f"{s_idx+1:02d}_{st.name}")
            _ensure_dir(stage_log_dir)
            try:
                # Interpolate sweep config first
                sweeps_cfg = _deep_interpolate(st.sweeps or {}, scope_vars)
                created = _generate_sweeps(repo_root, sweeps_cfg, scope_vars)
                # Write a small summary log
                summary_path = os.path.join(stage_log_dir, "stage.log")
                with open(summary_path, "w", encoding="utf-8") as lf:
                    lf.write(f"Generated {len(created)} sweep files:\n")
                    for p in created:
                        lf.write(p + "\n")
            except Exception as e:
                print(f"[orchestrator] make_sweeps failed: {e}")
                overall_rc = 1
                if stop_on_fail:
                    return overall_rc
            # proceed to next stage
            continue

        # Expand sweeps (cross-product across any sweep args)
        # Interpolate args/env with scope vars prior to sweep expansion
        stage_env = {k: _interpolate_string(v, scope_vars) for k, v in (st.env or {}).items()}
        stage_args_interpolated = _deep_interpolate(st.args or {}, scope_vars)

        expanded_runs = _expand_sweeps(repo_root, stage_args_interpolated)
        if not expanded_runs:
            expanded_runs = [stage_args_interpolated]

        for r_idx, args_map in enumerate(expanded_runs):
            # Compose command
            py = _venv_python(st.venv) if st.venv else default_python
            assert st.script is not None
            cmd: List[str] = [py, st.script]
            cmd.extend(_flatten_args(args_map))

            # Environment
            env = os.environ.copy()
            env.update(stage_env)
            env.setdefault("PIPELINE_STAGE_NAME", st.name)
            if env_name:
                env.setdefault("PIPELINE_ENV", env_name)
            # Surface core environment vars for convenience
            for core in ("ARTIFACTORY_BASE", "DATASET_BASE", "LOGS_BASE", "CONFIG_BASE", "WORK_BASE"):
                if core in scope_vars:
                    env.setdefault(f"PIPELINE_{core}", _interpolate_string(scope_vars[core], scope_vars))
                    # Also expose plain vars for config interpolation in train.py
                    env.setdefault(core, _interpolate_string(scope_vars[core], scope_vars))
            # Export all environment-config vars (e.g., custom PREPARED_DATASET) as plain env
            for k, v in (env_vars.items() if isinstance(env_vars, dict) else []):
                try:
                    env.setdefault(str(k), _interpolate_string(str(v), scope_vars))
                except Exception:
                    pass
            # Export all pipeline-level vars (e.g., MODEL_NAME, DATASET)
            for k, v in (pipeline_vars.items() if isinstance(pipeline_vars, dict) else []):
                try:
                    env.setdefault(str(k), _interpolate_string(str(v), scope_vars))
                except Exception:
                    pass
            env.setdefault("PIPELINE_TS", ts)
            env.setdefault("PIPELINE_RUN_DIR", base_log_dir)
            # Also expose repo/pipeline dirs for convenience
            env.setdefault("REPO_ROOT", repo_root)
            env.setdefault("PIPELINE_DIR", pipeline_dir)

            # Logs and stage/run context
            stage_token = f"{s_idx+1:02d}_{st.name}"
            stage_log_dir = os.path.join(base_log_dir, stage_token)
            _ensure_dir(stage_log_dir)
            # Add stage/run context as plain env vars for config interpolation
            env.setdefault("STAGE_NAME", st.name)
            env.setdefault("STAGE_INDEX", str(s_idx + 1))
            env.setdefault("RUN_INDEX", str(r_idx + 1))
            env.setdefault("RUN_DIR", base_log_dir)
            env.setdefault("STAGE_DIR", stage_log_dir)
            run_log = os.path.join(stage_log_dir, f"run_{r_idx+1:02d}.log") if len(expanded_runs) > 1 else os.path.join(stage_log_dir, "stage.log")

            rc = _run_subprocess(cmd, env=env, cwd=repo_root, log_path=run_log, dry_run=dry_run)
            if rc != 0:
                print(f"[orchestrator] Stage '{st.name}' run {r_idx+1} failed with exit code {rc}")
                overall_rc = rc
                if stop_on_fail:
                    return overall_rc
    return overall_rc


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a multi-stage pipeline from YAML")
    ap.add_argument("--pipeline", default=os.path.join("pipelines", "pipeline.yaml"), help="Path to pipeline YAML")
    ap.add_argument("--stop-on-fail", action=argparse.BooleanOptionalAction, default=True, help="Stop at first failing run")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    ap.add_argument("--log-dir", default=None, help="Base directory for logs (default: pipeline_runs/<ts>)")
    ap.add_argument("--stages", nargs="*", help="Only run these stages by name")
    ap.add_argument("--start-stage", default=None, help="Start running from this stage name (inclusive)")
    args = ap.parse_args()

    rc = run_pipeline(
        pipeline_yaml=args.pipeline,
        stop_on_fail=args.stop_on_fail,
        dry_run=bool(args.dry_run),
        log_dir=args.log_dir,
        stage_filter=set(args.stages) if args.stages else None,
        start_stage=args.start_stage,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
