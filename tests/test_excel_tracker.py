"""Regression tests for portable, append-only recording of experiment evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openpyxl import load_workbook

from experiment_tools.excel_tracker import WorkbookUpdateError, import_verified_run


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "output" / "experiment"
TEMPLATE = ROOT / "templates" / "ONNX_Experiment_Tracker.xlsx"
if not TEMPLATE.exists():
    TEMPLATE = ROOT / "outputs" / "01a18b4e-c25e-7641-a199-5f257667b937" / "ONNX_Experiment_Tracker.xlsx"


class ExcelTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.workbook = self.root / "records.xlsx"
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        for filename in (
            "status.json", "config.json", "dataset.json", "environment.json", "runtime_settings.json",
            "provenance.json", "summary.csv", "correctness.json",
        ):
            shutil.copy2(EVIDENCE / filename, self.run_dir / filename)
        (self.run_dir / "verification").mkdir()
        shutil.copy2(EVIDENCE / "verification" / "verification.json", self.run_dir / "verification" / "verification.json")
        # A distinct environment timestamp models an independently executed run,
        # even when all observed medians happen to be identical.
        environment = json.loads((self.run_dir / "environment.json").read_text(encoding="utf-8"))
        environment["recorded_at_utc"] = "2026-10-30T03:04:05+00:00"
        (self.run_dir / "environment.json").write_text(json.dumps(environment), encoding="utf-8")

    def test_import_populates_next_run_and_duplicate_does_not_rewrite(self) -> None:
        result = import_verified_run(self.run_dir, self.workbook, TEMPLATE, {
            "operator": "Teammate", "plugged_in": True, "power_mode": "Balanced", "notes": "Independent check",
        })
        self.assertTrue(result["imported"])
        self.assertEqual(result["run_id"], "RUN-002")
        book = load_workbook(self.workbook)
        try:
            self.assertEqual(book["Run Log"]["C8"].value, "Teammate")
            self.assertEqual(book["Run Log"]["G8"].value, "Yes")
            self.assertEqual(book["Run Log"]["B8"].value.hour, 11)
            self.assertEqual(book["Results"]["D32"].value, 1000)
            self.assertEqual(book["Results"]["E32"].value, 4.6844)
            self.assertEqual(book["Correctness"]["F11"].value, 0)
            self.assertIn("ISBLANK(F11)", book["Correctness"]["L11"].value)
            self.assertEqual(book["Run Log"]["C7"].value, "Not recorded")
            self.assertEqual(book["Results"]["E20"].value, 4.6844)
            self.assertTrue(book["Run Log"].column_dimensions["T"].hidden)
        finally:
            book.close()
        prior = self.workbook.read_bytes()
        repeated = import_verified_run(self.run_dir, self.workbook, TEMPLATE, {"operator": "Do not overwrite"})
        self.assertFalse(repeated["imported"])
        self.assertEqual(repeated["run_id"], "RUN-002")
        self.assertEqual(self.workbook.read_bytes(), prior)

    def test_independent_same_timings_receive_different_run_ids(self) -> None:
        import_verified_run(self.run_dir, self.workbook, TEMPLATE)
        environment = json.loads((self.run_dir / "environment.json").read_text(encoding="utf-8"))
        environment["recorded_at_utc"] = "2026-10-30T03:14:05+00:00"
        (self.run_dir / "environment.json").write_text(json.dumps(environment), encoding="utf-8")
        result = import_verified_run(self.run_dir, self.workbook, TEMPLATE)
        self.assertEqual(result["run_id"], "RUN-003")

    def test_long_run_metadata_wraps_without_resizing_existing_rows(self) -> None:
        template = load_workbook(TEMPLATE)
        try:
            previous_height = template["Run Log"].row_dimensions[7].height
        finally:
            template.close()
        long_note = "Power mode and background workload were recorded before timing. " * 12
        import_verified_run(self.run_dir, self.workbook, TEMPLATE, {
            "operator": "Independent reproduction team operator with a longer display name",
            "notes": long_note,
        })
        book = load_workbook(self.workbook)
        try:
            log = book["Run Log"]
            self.assertEqual(log["L8"].value, long_note)
            self.assertGreater(log.row_dimensions[8].height, 22)
            self.assertLessEqual(log.row_dimensions[8].height, 409)
            for column in ("C", "D", "H", "I", "J", "K", "L"):
                self.assertTrue(log[f"{column}8"].alignment.wrap_text)
            self.assertEqual(log.row_dimensions[7].height, previous_height)
        finally:
            book.close()

    def test_failed_or_corrupted_evidence_leaves_existing_workbook_unchanged(self) -> None:
        shutil.copy2(TEMPLATE, self.workbook)
        original_hash = hashlib.sha256(self.workbook.read_bytes()).hexdigest()
        for bad_value in ('"NaN"', '"0"'):
            with self.subTest(bad_value=bad_value):
                original = (self.run_dir / "summary.csv").read_text(encoding="utf-8")
                modified = original.replace("4.6844", bad_value, 1)
                (self.run_dir / "summary.csv").write_text(modified, encoding="utf-8")
                with self.assertRaises(ValueError):
                    import_verified_run(self.run_dir, self.workbook, TEMPLATE)
                self.assertEqual(hashlib.sha256(self.workbook.read_bytes()).hexdigest(), original_hash)
                (self.run_dir / "summary.csv").write_text(original, encoding="utf-8")
        (self.run_dir / "status.json").write_text('{"status":"failed"}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "complete"):
            import_verified_run(self.run_dir, self.workbook, TEMPLATE)
        self.assertEqual(hashlib.sha256(self.workbook.read_bytes()).hexdigest(), original_hash)

    def test_full_tracker_grows_without_losing_notes_settings_or_formulas(self) -> None:
        book = load_workbook(TEMPLATE)
        try:
            for row in range(8, 27):
                book["Run Log"].cell(row, 2, "Recorded historical row")
                book["Run Log"].cell(row, 12, f"Human note {row}")
            book["Run Log"]["L7"] = "Keep this manual note"
            book.save(self.workbook)
        finally:
            book.close()
        result = import_verified_run(self.run_dir, self.workbook, TEMPLATE)
        self.assertEqual(result["run_id"], "RUN-021")
        book = load_workbook(self.workbook)
        try:
            self.assertEqual(book["Run Log"].tables["RunConditions"].ref, "A6:L27")
            self.assertEqual(book["Run Log"].tables["RunSettings"].ref, "A33:Q54")
            self.assertEqual(book["Run Log"]["B34"].value, 42)
            self.assertEqual(book["Run Log"]["L7"].value, "Keep this manual note")
            self.assertEqual(book["Run Log"]["L26"].value, "Human note 26")
            self.assertEqual(book["Results"]["E260"].value, 4.6844)
            self.assertIn("$E$271", book["Results"]["J20"].value)
            self.assertIn("$A$34:$A$54", book["Correctness"]["L8"].value)
            self.assertEqual(book.defined_names["TrackerRunIDs"].attr_text, "'Run Log'!$A$7:$A$27")
            self.assertEqual(book["Results"]["A260"].value, "RUN-021")
            self.assertEqual(book["Correctness"]["A68"].value, "RUN-021")
        finally:
            book.close()

    def test_failed_destination_replace_keeps_recovery_and_original(self) -> None:
        shutil.copy2(TEMPLATE, self.workbook)
        original = self.workbook.read_bytes()
        with patch("experiment_tools.excel_tracker.os.replace", side_effect=PermissionError("Excel owns the destination")):
            with self.assertRaises(WorkbookUpdateError) as caught:
                import_verified_run(self.run_dir, self.workbook, TEMPLATE)
        self.assertEqual(self.workbook.read_bytes(), original)
        recovery = caught.exception.recovery_path
        self.assertIsNotNone(recovery)
        assert recovery is not None
        self.assertTrue(recovery.is_file())
        book = load_workbook(recovery)
        try:
            self.assertEqual(book["Results"]["A32"].value, "RUN-002")
            self.assertEqual(book["Results"]["E32"].value, 4.6844)
        finally:
            book.close()
        self.assertFalse((self.root / ".records.xlsx.lock").exists())

    def test_concurrent_import_lock_refuses_to_touch_destination(self) -> None:
        shutil.copy2(TEMPLATE, self.workbook)
        before = self.workbook.read_bytes()
        lock_file = self.root / ".records.xlsx.lock"
        lock_file.write_text('{"pid":1234}', encoding="utf-8")
        with self.assertRaisesRegex(WorkbookUpdateError, "Another importer"):
            import_verified_run(self.run_dir, self.workbook, TEMPLATE)
        self.assertEqual(self.workbook.read_bytes(), before)
        self.assertTrue(lock_file.exists())


if __name__ == "__main__":
    unittest.main()
