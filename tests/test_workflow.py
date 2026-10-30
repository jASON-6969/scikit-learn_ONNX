"""Test failure gates, logs and retry metadata independently of timing speed."""

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from experiment_tools.workflow import build_parser, execute_workflow, run_logged_command


class WorkflowTests(unittest.TestCase):
    def arguments(self, *arguments: str) -> Any:
        return build_parser().parse_args(list(arguments))

    def test_run_verifies_before_import_and_records_conditions(self) -> None:
        with tempfile.TemporaryDirectory(prefix="onnx team ") as temporary:
            root = Path(temporary)
            output = root / "output" / "new run"
            phases: list[str] = []

            def command(command: list[str], cwd: Path, console: Any) -> int:
                self.assertEqual(cwd, root)
                self.assertIn(str(output), command)
                stage = command[7]
                phases.append(stage)
                if stage == "run":
                    self.assertFalse(output.exists(), "Logger must not precreate the run directory")
                    output.mkdir(parents=True)
                else:
                    self.assertTrue((output / "run_conditions.json").exists())
                return 0

            def record(run: Path, workbook: Path, template: Path, metadata: dict[str, Any]) -> dict[str, Any]:
                phases.append("excel")
                self.assertEqual(metadata["operator"], "Team member")
                self.assertEqual(metadata["plugged_in"], "Yes")
                self.assertEqual(metadata["purpose"], "Full benchmark")
                self.assertIn("verification_finished_at_utc", metadata)
                return {"run_id": "RUN-002"}

            args = self.arguments("run", "--output", "output/new run", "--operator", "Team member", "--plugged-in", "Yes")
            with patch("experiment_tools.workflow.run_logged_command", side_effect=command), patch("experiment_tools.workflow.import_to_tracker", side_effect=record), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(execute_workflow(args, root), 0)
            self.assertEqual(phases, ["run", "verify", "excel"])
            log = next((root / "logs").glob("*/workflow.json"))
            saved = json.loads(log.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "complete")
            self.assertEqual(saved["excel_result"]["run_id"], "RUN-002")
            self.assertEqual(len(saved["steps"]), 6)

    def test_benchmark_failure_never_verifies_or_imports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.arguments("run", "--output", "output/failure")
            with patch("experiment_tools.workflow.run_logged_command", return_value=4) as commands, patch("experiment_tools.workflow.import_to_tracker") as importer, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(execute_workflow(args, root), 1)
            self.assertEqual(commands.call_count, 1)
            importer.assert_not_called()
            saved = json.loads(next((root / "logs").glob("*/workflow.json")).read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "benchmark_failed")
            self.assertIn("finished_at_utc", saved)

    def test_verification_failure_never_uses_stale_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output" / "existing"
            (output / "verification").mkdir(parents=True)
            (output / "verification" / "verification.json").write_text('{"passed":true}', encoding="utf-8")
            args = self.arguments("import", "--output", "output/existing")
            with patch("experiment_tools.workflow.run_logged_command", return_value=1), patch("experiment_tools.workflow.import_to_tracker") as importer, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(execute_workflow(args, root), 1)
            importer.assert_not_called()
            self.assertFalse((output / "run_conditions.json").exists())

    def test_existing_nonempty_directory_is_not_modified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "original"
            output.mkdir()
            marker = output / "evidence.txt"
            marker.write_text("keep", encoding="utf-8")
            args = self.arguments("run", "--output", "original")
            with patch("experiment_tools.workflow.run_logged_command") as commands, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(execute_workflow(args, root), 1)
            commands.assert_not_called()
            self.assertEqual(list(output.iterdir()), [marker])

    def test_retry_preserves_metadata_and_does_not_remeasure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "measured"
            output.mkdir()
            original = {"operator": "Original operator", "power_mode": "Balanced", "notes": "Original note", "purpose": "Independent reproduction"}
            (output / "run_conditions.json").write_text(json.dumps(original), encoding="utf-8")
            args = self.arguments("import", "--output", "measured", "--background-load", "Idle")
            with patch("experiment_tools.workflow.run_logged_command", return_value=0) as commands, patch("experiment_tools.workflow.import_to_tracker", return_value={"run_id": "RUN-002", "duplicate": True}) as importer, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(execute_workflow(args, root), 0)
            self.assertEqual(commands.call_count, 1)
            self.assertEqual(commands.call_args.args[0][7], "verify")
            recorded = importer.call_args.args[3]
            self.assertEqual(recorded["operator"], "Original operator")
            self.assertEqual(recorded["power_mode"], "Balanced")
            self.assertEqual(recorded["notes"], "Original note")
            self.assertEqual(recorded["background_load"], "Idle")

    def test_excel_failure_retains_verified_evidence_and_retry_command(self) -> None:
        with tempfile.TemporaryDirectory(prefix="team O'Brien ") as temporary:
            root = Path(temporary)
            output = root / "measured"
            output.mkdir()
            args = self.arguments("import", "--output", "measured")
            stderr = io.StringIO()
            with patch("experiment_tools.workflow.run_logged_command", return_value=0), patch("experiment_tools.workflow.import_to_tracker", side_effect=PermissionError("Close Excel")), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                self.assertEqual(execute_workflow(args, root), 1)
            self.assertIn("already verified", stderr.getvalue())
            self.assertIn("experiment_tools import", stderr.getvalue())
            self.assertIn("  & '", stderr.getvalue())
            self.assertIn("O''Brien", stderr.getvalue())
            self.assertTrue((output / "run_conditions.json").exists())
            saved = json.loads(next((root / "logs").glob("*/workflow.json")).read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "workflow_failed")

    def test_reduced_run_is_identified_as_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.arguments("run", "--warmup", "3", "--rounds", "2", "--calls", "5")
            with patch("experiment_tools.workflow.run_logged_command", return_value=0), patch("experiment_tools.workflow.import_to_tracker", return_value={"run_id": "RUN-002"}) as importer, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(execute_workflow(args, root), 0)
            self.assertEqual(importer.call_args.args[3]["purpose"], "Smoke test")

    def test_interruption_retains_log_and_releases_run_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.arguments("run", "--output", "new")
            with patch("experiment_tools.workflow.run_logged_command", side_effect=KeyboardInterrupt), patch("experiment_tools.workflow.import_to_tracker") as importer, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(execute_workflow(args, root), 130)
            importer.assert_not_called()
            saved = json.loads(next((root / "logs").glob("*/workflow.json")).read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "interrupted")
            self.assertEqual(list((root / "logs").glob(".run_*.lock")), [])

    def test_another_workflows_output_lock_is_never_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root / "logs"
            logs.mkdir()
            output = (root / "new").resolve()
            lock = logs / (".run_" + hashlib.sha256(str(output).encode("utf-8")).hexdigest() + ".lock")
            lock.write_text("another workflow", encoding="utf-8")
            args = self.arguments("run", "--output", "new")
            with patch("experiment_tools.workflow.run_logged_command") as commands, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(execute_workflow(args, root), 1)
            commands.assert_not_called()
            self.assertEqual(lock.read_text(encoding="utf-8"), "another workflow")

    def test_command_streaming_handles_spaces_without_a_shell(self) -> None:
        with tempfile.TemporaryDirectory(prefix="onnx team ") as temporary:
            console = io.StringIO()
            standard = io.StringIO()
            with contextlib.redirect_stdout(standard):
                code = run_logged_command([sys.executable, "-c", "print('one line with spaces')"], Path(temporary), console)
            self.assertEqual(code, 0)
            self.assertIn("one line with spaces", console.getvalue())
            self.assertIn("one line with spaces", standard.getvalue())


if __name__ == "__main__":
    unittest.main()
