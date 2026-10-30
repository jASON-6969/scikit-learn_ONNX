"""Execute, verify and record experiments with portable paths and durable logs."""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, TextIO


ROOT = Path(__file__).resolve().parents[1]
CONDITION_FIELDS = ("operator", "power_mode", "plugged_in", "background_load", "purpose", "notes")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_project_path(value: str | Path, root: Path) -> Path:
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else root / candidate).resolve()


def powershell_literal(value: str | Path) -> str:
    """Quote a recovery argument without interpreting spaces or shell syntax."""
    return "'" + str(value).replace("'", "''") + "'"


def save_json(path: Path, data: Any) -> None:
    """Replace status metadata atomically without truncating an earlier record."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_logged_command(command: list[str], root: Path, console: TextIO) -> int:
    """Stream the child output to the user and a UTF-8 file without a shell."""
    console.write("COMMAND " + json.dumps(command) + "\n")
    console.flush()
    child = subprocess.Popen(
        command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    try:
        assert child.stdout is not None
        for line in child.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            console.write(line)
            console.flush()
        return child.wait()
    except BaseException:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        raise
    finally:
        if child.stdout is not None:
            child.stdout.close()


def import_to_tracker(
    run_dir: Path, workbook_path: Path, template_path: Path, metadata: dict[str, Any],
) -> dict[str, Any]:
    # Help and phase logs work even when optional workbook dependencies are absent.
    from .excel_tracker import import_verified_run

    return import_verified_run(run_dir, workbook_path, template_path, metadata)


def metadata_for_run(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    existing = output / "run_conditions.json"
    metadata: dict[str, Any] = {}
    if args.command == "import" and existing.exists():
        value = json.loads(existing.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("run_conditions.json must contain an object")
        metadata.update(value)
    for field in CONDITION_FIELDS:
        value = getattr(args, field)
        if value is not None:
            metadata[field] = value
        elif field not in metadata:
            metadata[field] = "" if field == "notes" else "Not recorded"
    if args.command == "run" and args.purpose is None:
        metadata["purpose"] = "Full benchmark" if (args.warmup, args.rounds, args.calls) == (50, 10, 100) else "Smoke test"
    return metadata


def execute_workflow(args: argparse.Namespace, root: Path = ROOT) -> int:
    """A failed measurement or fresh verification never reaches the importer."""
    root = root.resolve()
    workflow_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    log_dir = root / "logs" / workflow_id
    log_dir.mkdir(parents=True, exist_ok=False)
    output = resolve_project_path(args.output or f"output/runs/{workflow_id}", root)
    workbook_path = resolve_project_path(args.workbook, root)
    template_path = root / "templates" / "ONNX_Experiment_Tracker.xlsx"
    state: dict[str, Any] = {
        "workflow_id": workflow_id, "command": args.command,
        "started_at_utc": utc_now(), "status": "starting",
        "source_directory": str(output), "workbook_path": str(workbook_path),
        "workflow_log_directory": str(log_dir), "steps": [],
    }
    metadata: dict[str, Any] = {}
    can_write_run_metadata = False
    output_lock = root / "logs" / (".run_" + hashlib.sha256(str(output).encode("utf-8")).hexdigest() + ".lock")
    lock_owned = False

    def record_step(stage: str, phase_status: str, **details: Any) -> None:
        event = {"time_utc": utc_now(), "stage": stage, "status": phase_status, **details}
        state["steps"].append(event)
        state["status"] = phase_status if phase_status.endswith("failed") else stage
        if phase_status.endswith("failed"):
            state["finished_at_utc"] = utc_now()
        save_json(log_dir / "workflow.json", state)
        with (log_dir / "events.jsonl").open("a", encoding="utf-8") as events:
            events.write(json.dumps(event, allow_nan=False) + "\n")

    def save_conditions() -> None:
        if can_write_run_metadata and output.is_dir():
            save_json(output / "run_conditions.json", metadata)

    try:
        with output_lock.open("x", encoding="utf-8") as lock_stream:
            lock_stream.write(f"workflow_id={workflow_id}\noutput={output}\n")
        lock_owned = True
        if args.command == "run":
            if args.warmup < 0 or args.rounds <= 0 or args.calls <= 0:
                raise ValueError("warmup must be nonnegative; rounds and calls must be positive")
            if output.exists() and (not output.is_dir() or any(output.iterdir())):
                raise ValueError("Output directory is not empty; choose a new path")
        elif not output.is_dir():
            raise ValueError("Import requires an existing experiment output directory")
        metadata = metadata_for_run(args, output)
        metadata.update({
            "workflow_id": workflow_id, "workflow_log_directory": str(log_dir),
            "console_log": str(log_dir / "console.log"), "source_directory": str(output),
            "workbook_path": str(workbook_path),
        })
        state["hostname"] = socket.gethostname()
        if args.command == "run":
            metadata["hostname"] = socket.gethostname()
            metadata["measurement_workflow_id"] = workflow_id
            metadata["measurement_console_log"] = str(log_dir / "console.log")
        else:
            metadata.setdefault("hostname", "Not recorded")
            metadata["import_hostname"] = socket.gethostname()
        state["metadata"] = metadata
        save_json(log_dir / "workflow.json", state)
        print(f"Workflow log: {log_dir}", flush=True)
        print(f"Experiment output: {output}", flush=True)
        with (log_dir / "console.log").open("w", encoding="utf-8") as console:
            prefix = [sys.executable, "-B", "-X", "utf8", "-u", "-m", "onnx_benchmark"]
            if args.command == "run":
                command = [*prefix, "run", "--output", str(output), "--warmup", str(args.warmup), "--rounds", str(args.rounds), "--calls", str(args.calls)]
                metadata["run_started_at_utc"] = utc_now()
                can_write_run_metadata = True
                record_step("benchmark", "running", command=command)
                return_code = run_logged_command(command, root, console)
                metadata["run_finished_at_utc"] = utc_now()
                save_conditions()
                record_step("benchmark", "complete" if return_code == 0 else "benchmark_failed", return_code=return_code)
                if return_code:
                    return 1
            command = [*prefix, "verify", "--output", str(output)]
            record_step("verification", "running", command=command)
            return_code = run_logged_command(command, root, console)
            record_step("verification", "complete" if return_code == 0 else "verification_failed", return_code=return_code)
            if return_code:
                return 1
            metadata["verification_finished_at_utc"] = utc_now()
            can_write_run_metadata = True
            save_conditions()
            record_step("excel_import", "running")
            result = import_to_tracker(output, workbook_path, template_path, metadata)
            state["excel_result"] = result
            record_step("excel_import", "complete", result=result)
            state["status"] = "complete"
            state["finished_at_utc"] = utc_now()
            save_json(log_dir / "workflow.json", state)
            if result.get("imported") is False:
                print(f"Excel already contains this evidence: {workbook_path} ({result.get('run_id')}); no duplicate added.", flush=True)
            else:
                print(f"Excel recorded: {workbook_path} ({result.get('run_id', 'recorded')})", flush=True)
            print("Workflow complete.", flush=True)
            return 0
    except KeyboardInterrupt:
        metadata["interrupted_at_utc"] = utc_now()
        save_conditions()
        record_step("workflow", "interrupted")
        state["status"] = "interrupted"
        state["finished_at_utc"] = utc_now()
        save_json(log_dir / "workflow.json", state)
        print(f"Interrupted. Evidence and logs were retained: {log_dir}", file=sys.stderr)
        return 130
    except Exception as error:
        record_step("workflow", "workflow_failed", error=str(error), error_type=type(error).__name__)
        state["finished_at_utc"] = utc_now()
        save_json(log_dir / "workflow.json", state)
        print(f"Error: {error}", file=sys.stderr)
        print(f"Log: {log_dir}", file=sys.stderr)
        if metadata.get("verification_finished_at_utc"):
            print("The experiment is already verified. Close Excel and retry with:", file=sys.stderr)
            print(f"  & {powershell_literal(sys.executable)} -B -X utf8 -m experiment_tools import --output {powershell_literal(output)} --workbook {powershell_literal(workbook_path)}", file=sys.stderr)
        return 1
    finally:
        if lock_owned:
            output_lock.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run, freshly verify and automatically record an experiment in Excel.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description in (("run", "Run the benchmark, verify it, then record it"), ("import", "Freshly verify an existing run, then record it without remeasuring")):
        command = commands.add_parser(name, help=description)
        command.add_argument("--output", required=name == "import", help="Experiment directory; relative paths are resolved from the project root")
        command.add_argument("--workbook", default="ONNX_Experiment_Tracker.xlsx", help="Tracker path; created from the bundled template when absent")
        command.add_argument("--operator", help="Person who performed the measurement")
        command.add_argument("--plugged-in", choices=("Yes", "No", "Not recorded"))
        command.add_argument("--power-mode", help="Actual OS power mode during measurement")
        command.add_argument("--background-load", help="Other CPU workloads present during measurement")
        command.add_argument("--purpose", help="Full benchmark, independent reproduction, smoke test, etc.")
        command.add_argument("--notes", help="Additional measurement conditions")
        command.set_defaults(warmup=50, rounds=10, calls=100)
        if name == "run":
            command.add_argument("--warmup", type=int, default=50)
            command.add_argument("--rounds", type=int, default=10)
            command.add_argument("--calls", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return execute_workflow(build_parser().parse_args(argv))
