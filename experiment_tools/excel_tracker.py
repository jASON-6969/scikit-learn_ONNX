"""Append verified experiment evidence to the team's portable Excel tracker.

This module deliberately lives outside ``onnx_benchmark``: recording results must
not change the measured package's provenance hashes. Excel is not required.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
import unicodedata
from contextlib import contextmanager
from copy import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from openpyxl import load_workbook
from openpyxl.workbook.defined_name import DefinedName
from openpyxl.workbook.properties import CalcProperties
from openpyxl.worksheet.cell_range import CellRange
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.worksheet import Worksheet


BACKENDS = ("sklearn", "onnx_zipmap", "onnx_tensor")
EVIDENCE_FILES = (
    "config.json", "dataset.json", "environment.json", "provenance.json",
    "runtime_settings.json", "summary.csv", "correctness.json",
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
HKT = timezone(timedelta(hours=8))


class WorkbookUpdateError(OSError):
    """Updating the destination failed; a recoverable workbook may be available."""

    def __init__(self, message: str, recovery_path: Path | None = None) -> None:
        super().__init__(message)
        self.recovery_path = recovery_path


def _json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return result


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Expected a numeric {label}")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Expected a numeric {label}") from error
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"Invalid {label}: {value}")
    return result


def _integer(value: Any, label: str) -> int:
    result = _number(value, label, positive=True)
    if not result.is_integer():
        raise ValueError(f"Expected an integer {label}")
    return int(result)


def _identity(run_dir: Path) -> str:
    """Include the environment timestamp, so independent runs stay distinct."""
    digest = hashlib.sha256()
    for filename in EVIDENCE_FILES:
        digest.update(filename.encode("utf-8") + b"\0")
        data = (run_dir / filename).read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _read_evidence(run_dir: Path) -> dict[str, Any]:
    status = _json(run_dir / "status.json")
    verified = _json(run_dir / "verification" / "verification.json")
    if status.get("status") != "complete" or verified.get("passed") is not True:
        raise ValueError("Only a complete, successfully verified run can be imported")
    config = _json(run_dir / "config.json")
    dataset = _json(run_dir / "dataset.json")
    environment = _json(run_dir / "environment.json")
    correctness = _json(run_dir / "correctness.json")
    runtime = _json(run_dir / "runtime_settings.json")
    batches = tuple(_integer(value, "batch size") for value in config["batch_sizes"])
    if batches != (1, 16, 64, 256):
        raise ValueError("This tracker requires batch sizes 1, 16, 64, 256")
    calls = _integer(config["rounds"], "rounds") * _integer(config["calls_per_round"], "calls per round")
    expected = {(backend, batch) for batch in batches for backend in BACKENDS}
    records: dict[tuple[str, int], dict[str, Any]] = {}
    with (run_dir / "summary.csv").open(encoding="utf-8", newline="") as stream:
        for item in csv.DictReader(stream):
            key = (item["variant"], _integer(item["batch_size"], "batch size"))
            if key not in expected or key in records:
                raise ValueError(f"Missing, unexpected or duplicate performance key: {key}")
            record: dict[str, Any] = {"calls": _integer(item["calls"], "timed calls")}
            for field in ("median_ms", "p25_ms", "p75_ms", "p95_ms", "throughput_samples_per_second", "speedup_vs_sklearn"):
                record[field] = _number(item[field], field, positive=True)
            if record["calls"] != calls:
                raise ValueError("Summary call counts differ from the run configuration")
            if not record["p25_ms"] <= record["median_ms"] <= record["p75_ms"] <= record["p95_ms"]:
                raise ValueError("Summary quantiles are not ordered")
            records[key] = record
    if set(records) != expected:
        raise ValueError("The summary does not contain all 12 configurations")
    if verified.get("groups") != len(expected) or verified.get("timing_rows") != calls * len(expected):
        raise ValueError("Saved verification counts differ from the imported summary")
    tolerance = _number(config["probability_tolerance"], "probability tolerance", positive=True)
    test_count = _integer(correctness["test_count"], "held-out cases")
    if test_count != dataset["test_count"] or correctness.get("passed") is not True:
        raise ValueError("Held-out correctness has not passed")
    if set(correctness.get("variants", {})) != set(BACKENDS):
        raise ValueError("Missing or unexpected correctness backends")
    for backend in BACKENDS:
        item = correctness["variants"][backend]
        accuracy = _number(item["accuracy"], "accuracy")
        agreement = _number(item["label_agreement"], "label agreement")
        error = _number(item["max_probability_abs_error"], "probability error")
        if not 0 <= accuracy <= 1 or agreement != 1 or not 0 <= error <= tolerance:
            raise ValueError(f"Invalid correctness metrics: {backend}")
        for flag in ("passed", "probabilities_finite", "probabilities_bounded", "probability_rows_normalized"):
            if item.get(flag) is not True:
                raise ValueError(f"Failed correctness flag: {backend}/{flag}")
        batch_checks = correctness.get("performance_batch_checks", {}).get(backend, {})
        if set(batch_checks) != {str(batch) for batch in batches} or any(
            item.get("passed") is not True for item in batch_checks.values()
        ):
            raise ValueError(f"Performance-batch correctness has not passed: {backend}")
    for (backend, batch), record in records.items():
        throughput = batch * 1000 / record["median_ms"]
        speedup = records[("sklearn", batch)]["median_ms"] / record["median_ms"]
        if not math.isclose(record["throughput_samples_per_second"], throughput, rel_tol=1e-10):
            raise ValueError("Saved throughput disagrees with median latency")
        if not math.isclose(record["speedup_vs_sklearn"], speedup, rel_tol=1e-10):
            raise ValueError("Saved speedup disagrees with the matching baseline")
    captured = datetime.fromisoformat(environment["recorded_at_utc"])
    if captured.tzinfo is None:
        raise ValueError("Environment timestamp must include its timezone")
    return {
        "config": config, "dataset": dataset, "environment": environment,
        "correctness": correctness, "runtime": runtime, "records": records,
        "captured_hkt": captured.astimezone(HKT).replace(tzinfo=None),
        "evidence_id": _identity(run_dir),
    }


@contextmanager
def _workbook_lock(workbook_path: Path) -> Iterator[None]:
    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = workbook_path.with_name(f".{workbook_path.name}.lock")
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise WorkbookUpdateError(
            f"Another importer is updating this workbook, or a previous importer was interrupted. "
            f"Do not start another import. If no importer is running, remove the stale lock: {lock_path}"
        ) from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "created_at_utc": datetime.now(timezone.utc).isoformat()}, stream)
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _bounds(sheet: Worksheet, table_name: str) -> CellRange:
    if table_name not in sheet.tables:
        raise ValueError(f"Tracker table is missing: {table_name}")
    return CellRange(sheet.tables[table_name].ref)


def _copy_row(sheet: Worksheet, source: int, destination: int, width: int) -> None:
    for column in range(1, width + 1):
        old = sheet.cell(source, column)
        new = sheet.cell(destination, column)
        new._style = copy(old._style)
        new.alignment = copy(old.alignment)
        new.protection = copy(old.protection)
        new.number_format = old.number_format
    sheet.row_dimensions[destination].height = sheet.row_dimensions[source].height


def _extend_slot(workbook: Any) -> str:
    runs = workbook["Run Log"]
    condition = _bounds(runs, "RunConditions")
    settings = _bounds(runs, "RunSettings")
    numbers = [int(str(runs.cell(row, 1).value).split("-")[-1]) for row in range(condition.min_row + 1, condition.max_row + 1)]
    run_id = f"RUN-{max(numbers) + 1:03d}"
    insertion = condition.max_row + 1
    runs.insert_rows(insertion)
    _copy_row(runs, insertion - 1, insertion, 12)
    runs.cell(insertion, 1, run_id)
    runs.tables["RunConditions"].ref = f"A{condition.min_row}:L{insertion}"
    runs.tables["RunConditions"].autoFilter.ref = runs.tables["RunConditions"].ref
    settings.min_row += 1
    settings.max_row += 1
    new_setting_row = settings.max_row + 1
    _copy_row(runs, settings.max_row, new_setting_row, 17)
    runs.cell(new_setting_row, 1, run_id)
    runs.tables["RunSettings"].ref = f"A{settings.min_row}:Q{new_setting_row}"
    runs.tables["RunSettings"].autoFilter.ref = runs.tables["RunSettings"].ref
    results = workbook["Results"]
    performance = _bounds(results, "PerformanceRecords")
    for index, (batch, backend) in enumerate((batch, backend) for batch in (1, 16, 64, 256) for backend in BACKENDS):
        row = performance.max_row + index + 1
        _copy_row(results, performance.max_row - 11 + index, row, 13)
        for column, value in enumerate((run_id, backend, batch), 1):
            results.cell(row, column, value)
    results.tables["PerformanceRecords"].ref = f"A{performance.min_row}:M{performance.max_row + 12}"
    results.tables["PerformanceRecords"].autoFilter.ref = results.tables["PerformanceRecords"].ref
    correctness = workbook["Correctness"]
    accuracy = _bounds(correctness, "CorrectnessRecords")
    for index, backend in enumerate(BACKENDS):
        row = accuracy.max_row + index + 1
        _copy_row(correctness, accuracy.max_row - 2 + index, row, 13)
        correctness.cell(row, 1, run_id)
        correctness.cell(row, 2, backend)
    correctness.tables["CorrectnessRecords"].ref = f"A{accuracy.min_row}:M{accuracy.max_row + 3}"
    correctness.tables["CorrectnessRecords"].autoFilter.ref = correctness.tables["CorrectnessRecords"].ref
    return run_id


def _row_for(sheet: Worksheet, table_name: str, run_id: str) -> int:
    limits = _bounds(sheet, table_name)
    matches = [row for row in range(limits.min_row + 1, limits.max_row + 1) if sheet.cell(row, 1).value == run_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one {run_id} row in {table_name}, found {len(matches)}")
    return matches[0]


def _slot_is_empty(workbook: Any, run_id: str) -> bool:
    runs = workbook["Run Log"]
    row = _row_for(runs, "RunConditions", run_id)
    # A planned operator/purpose is allowed; an existing evidence record is not.
    if runs.cell(row, 2).value is not None or runs.cell(row, 11).value not in (None, ""):
        return False
    for sheet_name, table_name, columns in (
        ("Results", "PerformanceRecords", range(4, 9)),
        ("Correctness", "CorrectnessRecords", range(3, 12)),
    ):
        sheet = workbook[sheet_name]
        limits = _bounds(sheet, table_name)
        for data_row in range(limits.min_row + 1, limits.max_row + 1):
            if sheet.cell(data_row, 1).value == run_id and any(sheet.cell(data_row, column).value is not None for column in columns):
                return False
    return True


def _source_display(run_dir: Path) -> str:
    try:
        return run_dir.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(run_dir)


def _registry(workbook: Any, evidence_id: str) -> str | None:
    runs = workbook["Run Log"]
    headers = ("Run ID", "Evidence SHA256", "Imported at UTC", "Source directory", "Import metadata JSON")
    for column, value in enumerate(headers, 19):
        runs.cell(6, column, value)
        runs.column_dimensions[runs.cell(6, column).column_letter].hidden = True
    limits = _bounds(runs, "RunConditions")
    for row in range(limits.min_row + 1, limits.max_row + 1):
        run_id = str(runs.cell(row, 1).value)
        recorded_id = runs.cell(row, 20).value
        if not recorded_id:
            recorded_path = runs.cell(row, 11).value
            if recorded_path:
                candidate = Path(str(recorded_path))
                if not candidate.is_absolute():
                    candidate = PROJECT_ROOT / candidate
                try:
                    recorded_id = _identity(candidate)
                except OSError:
                    # For a moved bundle, source paths may refer to a teammate's machine.
                    # Do not infer identity from a matching directory name alone.
                    recorded_id = None
                if recorded_id:
                    runs.cell(row, 19, run_id)
                    runs.cell(row, 20, recorded_id)
                    runs.cell(row, 22, str(candidate))
        if recorded_id == evidence_id:
            return run_id
    return None


def _set_values(sheet: Worksheet, row: int, values: list[Any]) -> None:
    for column, value in enumerate(values, 1):
        # Human text must never become a workbook formula.
        cell = sheet.cell(row, column)
        cell.value = value
        if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
            cell.data_type = "s"


def _fit_run_conditions(sheet: Worksheet, row: int) -> None:
    """Keep variable-length human metadata readable without resizing other rows."""
    required_height = 22.0
    for column in (3, 4, 8, 9, 10, 11, 12):
        cell = sheet.cell(row, column)
        alignment = copy(cell.alignment)
        alignment.wrap_text = True
        alignment.vertical = "top"
        cell.alignment = alignment
        text = str(cell.value or "").expandtabs(4)
        font_size = float(cell.font.sz or 10)
        # Excel column widths are approximate character units. Reserve a margin
        # for cell padding and proportional glyphs, including wide user names.
        width = max(1.0, float(sheet.column_dimensions[cell.column_letter].width or 13) - 2)
        line_capacity = max(1.0, width * 0.85 * 11 / font_size)
        lines = 0
        for paragraph in text.split("\n"):
            units = sum(2 if unicodedata.east_asian_width(character) in ("F", "W") else 1 for character in paragraph)
            lines += max(1, math.ceil(units / line_capacity))
        required_height = max(required_height, lines * (font_size + 4) + 6)
    # Excel supports at most 409 points for a row. Full text remains in the cell
    # and in the run's JSON/console logs even for an unusually long note.
    sheet.row_dimensions[row].height = min(409.0, required_height)


def _fill_run(workbook: Any, run_id: str, run_dir: Path, data: dict[str, Any], metadata: dict[str, Any]) -> None:
    runs = workbook["Run Log"]
    row = _row_for(runs, "RunConditions", run_id)
    config, dataset, environment = data["config"], data["dataset"], data["environment"]
    full = config["warmup_calls"] == 50 and config["rounds"] == 10 and config["calls_per_round"] == 100
    purpose = str(metadata.get("purpose") or ("Full benchmark" if full else "Smoke test"))
    processor = environment.get("cpu", {}).get("windows_processors", {})
    if isinstance(processor, list):
        cpu_name = "; ".join(str(item.get("Name", "" )).strip() for item in processor)
    else:
        cpu_name = str(processor.get("Name") or environment.get("cpu", {}).get("name") or "Not recorded").strip()
    plugged_in = metadata.get("plugged_in")
    if isinstance(plugged_in, bool):
        plugged_in = "Yes" if plugged_in else "No"
    supplied = [
        run_id, data["captured_hkt"], metadata.get("operator") or "Not recorded", purpose,
        "Complete", "Passed", plugged_in or "Not recorded", metadata.get("power_mode") or "Not recorded",
        metadata.get("background_load") or "Not recorded", cpu_name, _source_display(run_dir), metadata.get("notes") or "Imported after successful verification",
    ]
    # Preserve pre-entered human conditions when no replacement was supplied.
    key_columns = {3: "operator", 4: "purpose", 7: "plugged_in", 8: "power_mode", 9: "background_load", 12: "notes"}
    for column, key in key_columns.items():
        if not metadata.get(key) and runs.cell(row, column).value not in (None, ""):
            supplied[column - 1] = runs.cell(row, column).value
    _set_values(runs, row, supplied)
    _fit_run_conditions(runs, row)
    runs.cell(row, 2).number_format = "yyyy-mm-dd hh:mm:ss"
    packages = environment["packages"]
    setting_row = _row_for(runs, "RunSettings", run_id)
    _set_values(runs, setting_row, [
        run_id, config["seed"], config["n_estimators"], config["warmup_calls"], config["rounds"], config["calls_per_round"],
        dataset["train_count"], dataset["test_count"], data["runtime"]["sklearn"]["native_threadpools"],
        config["probability_tolerance"], environment["python"].split()[0], packages["scikit-learn"], packages["skl2onnx"],
        packages["onnx"], packages["onnxruntime"], dataset["dtype"], dataset["name"],
    ])
    results = workbook["Results"]
    bounds = _bounds(results, "PerformanceRecords")
    seen: set[tuple[str, int]] = set()
    for result_row in range(bounds.min_row + 1, bounds.max_row + 1):
        if results.cell(result_row, 1).value != run_id:
            continue
        performance_key = (str(results.cell(result_row, 2).value), int(results.cell(result_row, 3).value))
        if performance_key in seen or performance_key not in data["records"]:
            raise ValueError("The tracker has duplicate or invalid performance slots")
        seen.add(performance_key)
        record = data["records"][performance_key]
        for column, value in enumerate([record["calls"], record["median_ms"], record["p25_ms"], record["p75_ms"], record["p95_ms"]], 4):
            results.cell(result_row, column, value)
        results.cell(result_row, 13, f"Verified import. {purpose}.")
    if len(seen) != 12:
        raise ValueError("The tracker does not have exactly 12 performance slots for this run")
    correctness = workbook["Correctness"]
    bounds = _bounds(correctness, "CorrectnessRecords")
    seen_backends: set[str] = set()
    for correctness_row in range(bounds.min_row + 1, bounds.max_row + 1):
        if correctness.cell(correctness_row, 1).value != run_id:
            continue
        backend = str(correctness.cell(correctness_row, 2).value)
        if backend in seen_backends or backend not in BACKENDS:
            raise ValueError("The tracker has duplicate or invalid correctness slots")
        seen_backends.add(backend)
        item = data["correctness"]["variants"][backend]
        for column, value in enumerate([
            data["correctness"]["test_count"], item["accuracy"], item["label_agreement"], item["max_probability_abs_error"],
            "Pass", "Pass", "Pass", "Pass", "Pass",
        ], 3):
            correctness.cell(correctness_row, column, value)
        correctness.cell(correctness_row, 13, "Verified original held-out prediction checks")
    if seen_backends != set(BACKENDS):
        raise ValueError("The tracker does not have three correctness slots for this run")
    for column, value in enumerate([
        run_id, data["evidence_id"], datetime.now(timezone.utc).isoformat(), str(run_dir),
        json.dumps(metadata, ensure_ascii=True, default=str),
    ], 19):
        runs.cell(row, column, value)


def _refresh_formulas(workbook: Any) -> None:
    results = workbook["Results"]
    runs = workbook["Run Log"]
    correctness = workbook["Correctness"]
    perf = _bounds(results, "PerformanceRecords")
    settings = _bounds(runs, "RunSettings")
    conditions = _bounds(runs, "RunConditions")
    first, last = perf.min_row + 1, perf.max_row
    def bounded(column: str) -> str:
        return f"${column}${first}:${column}${last}"
    def matching(run: str, backend: str, batch: str) -> str:
        return f"{bounded('A')},{run},{bounded('B')},{backend},{bounded('C')},{batch}"
    def count(run: str, backend: str, batch: str) -> str:
        return f'COUNTIFS({matching(run, backend, batch)},{bounded("E")},">0")'
    def summed(run: str, backend: str, batch: str, column: str = "E") -> str:
        return f"SUMIFS({bounded(column)},{matching(run, backend, batch)})"
    for row in range(first, last + 1):
        baseline_count = count(f"$A{row}", '"sklearn"', f"$C{row}")
        baseline = summed(f"$A{row}", '"sklearn"', f"$C{row}")
        zip_count = count(f"$A{row}", '"onnx_zipmap"', f"$C{row}")
        zip_median = summed(f"$A{row}", '"onnx_zipmap"', f"$C{row}")
        formulas = [
            f'=IF(ISBLANK(E{row}),"",IF(OR(E{row}<=0,C{row}<=0),"Invalid input",C{row}*1000/E{row}))',
            f'=IF(ISBLANK(E{row}),"",IF(E{row}<=0,"Invalid latency",IF({baseline_count}=1,{baseline}/E{row},"Missing/duplicate baseline")))',
            f'=IF(OR(B{row}<>"onnx_tensor",ISBLANK(E{row})),"",IF(E{row}<=0,"Invalid latency",IF({zip_count}=1,1-E{row}/{zip_median},"Missing/duplicate ZipMap")))',
            f'=IF(COUNT(D{row}:H{row})=0,"Pending input",IF(COUNT(D{row}:H{row})<>5,"Incomplete input",IF(OR(D{row}<=0,D{row}<>INT(D{row}),E{row}<=0,F{row}<=0,F{row}>E{row},E{row}>G{row},G{row}>H{row}),"Invalid input",IF(COUNTIFS({matching(f"A{row}", f"B{row}", f"C{row}")})<>1,"Duplicate key","Recorded"))))',
        ]
        for column, value in enumerate(formulas, 9):
            results.cell(row, column, value)
    for row in range(7, 11):
        for column, backend in enumerate(BACKENDS, 2):
            counter = count('$B$4', f'"{backend}"', f'$A{row}')
            total = summed('$B$4', f'"{backend}"', f'$A{row}')
            results.cell(row, column, f'=IF({counter}=0,"",IF({counter}=1,{total},"Duplicate key"))')
        for destination, source, prerequisite in ((5, "J", "B"), (6, "K", "C")):
            total = summed('$B$4', '"onnx_tensor"', f'$A{row}', source)
            results.cell(row, destination, f'=IF(OR({prerequisite}{row}="",D{row}=""),"",IF(AND(ISNUMBER({prerequisite}{row}),ISNUMBER(D{row})),{total},"Check inputs"))')
    accuracy = _bounds(correctness, "CorrectnessRecords")
    settings_first, settings_last = settings.min_row + 1, settings.max_row
    for row in range(accuracy.min_row + 1, accuracy.max_row + 1):
        ids = f"'Run Log'!$A${settings_first}:$A${settings_last}"
        tolerances = f"'Run Log'!$J${settings_first}:$J${settings_last}"
        counter = f'COUNTIFS({ids},A{row},{tolerances},">0")'
        tolerance = f"SUMIFS({tolerances},{ids},A{row})"
        correctness.cell(row, 12, f'=IF(ISBLANK(F{row}),"",IF(F{row}<0,"Invalid input",IF({counter}<>1,"Missing/duplicate tolerance",IF(F{row}<={tolerance},"Pass","Fail"))))')
    workbook.defined_names.add(DefinedName("TrackerRunIDs", attr_text=f"'Run Log'!$A${conditions.min_row + 1}:$A${conditions.max_row}"))
    # Replace only the template-owned validations, leaving unrelated custom rules.
    for sheet in (results, runs, correctness):
        sheet.data_validations.dataValidation = [rule for rule in sheet.data_validations.dataValidation if rule.type != "list"]
    def dropdown(sheet: Worksheet, address: str, formula: str) -> None:
        rule = DataValidation(type="list", formula1=formula, allow_blank=True)
        rule.errorTitle = "Choose a listed value"
        rule.error = "Use one of the values in the list."
        rule.showErrorMessage = True
        sheet.add_data_validation(rule)
        rule.add(address)
    dropdown(results, "B4", "TrackerRunIDs")
    dropdown(results, f"A{first}:A{last}", "TrackerRunIDs")
    dropdown(results, f"B{first}:B{last}", '"sklearn,onnx_zipmap,onnx_tensor"')
    for column_name, values in (
        ("D", "Full benchmark,Controlled repeat,Smoke test,Independent reproduction,Workflow validation,Modified configuration"),
        ("E", "Planned,Running,Complete,Failed,Interrupted"),
        ("F", "Not run,Passed,Failed,Not recorded"),
        ("G", "Yes,No,Not recorded"),
    ):
        dropdown(runs, f"{column_name}{conditions.min_row + 1}:{column_name}{conditions.max_row}", f'"{values}"')
    dropdown(correctness, f"G{accuracy.min_row + 1}:K{accuracy.max_row}", '"Pass,Fail,Not recorded"')
    # A grown table must keep its existing dynamic warning formatting. Rule
    # formulas stay anchored to their original first data row.
    for sheet, target_first, target_last in (
        (results, first, last),
        (runs, conditions.min_row + 1, conditions.max_row),
        (correctness, accuracy.min_row + 1, accuracy.max_row),
    ):
        changes = []
        for conditional in sheet.conditional_formatting:
            ranges = list(conditional.sqref.ranges)
            if len(ranges) != 1 or ranges[0].min_row != target_first:
                continue
            current = ranges[0]
            if current.max_row != target_last:
                extended = CellRange(str(current))
                extended.max_row = target_last
                changes.append((str(conditional.sqref), str(extended), [copy(rule) for rule in conditional.rules]))
        for old_range, new_range, rules in changes:
            del sheet.conditional_formatting[old_range]
            for rule in rules:
                sheet.conditional_formatting.add(new_range, rule)
    workbook.calculation = CalcProperties(calcId=191029, fullCalcOnLoad=True, forceFullCalc=True, calcMode="auto")


def _atomic_save(workbook: Any, workbook_path: Path) -> None:
    descriptor, filename = tempfile.mkstemp(prefix=f"{workbook_path.stem}.recovery_", suffix=".xlsx", dir=workbook_path.parent)
    os.close(descriptor)
    temporary = Path(filename)
    try:
        workbook.save(temporary)
        check = load_workbook(temporary, read_only=True)
        try:
            if check.sheetnames != ["Results", "Run Log", "Correctness", "Guide"]:
                raise ValueError("Unexpected sheets in the saved tracker")
        finally:
            check.close()
        try:
            os.replace(temporary, workbook_path)
        except OSError as error:
            raise WorkbookUpdateError(
                f"Could not replace {workbook_path}. Close Excel, then retry the import. "
                f"A complete recovery workbook is saved at {temporary}. The original was not replaced.",
                recovery_path=temporary,
            ) from error
    except WorkbookUpdateError:
        raise
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def import_verified_run(
    run_dir: Path, workbook_path: Path, template_path: Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Import a freshly verified run, or return its existing RUN ID unchanged.

    The caller must run the benchmark's verifier immediately before calling this
    function. Saved verification JSON alone cannot prove that verification was
    performed in the current invocation. This function also checks completeness,
    numeric invariants, and saved counts before touching the destination.
    """
    run_dir, workbook_path, template_path = run_dir.resolve(), workbook_path.resolve(), template_path.resolve()
    data = _read_evidence(run_dir)
    if not workbook_path.exists() and not template_path.is_file():
        raise FileNotFoundError(f"Tracker template not found: {template_path}")
    with _workbook_lock(workbook_path):
        workbook = load_workbook(workbook_path if workbook_path.exists() else template_path)
        try:
            for name in ("Results", "Run Log", "Correctness", "Guide"):
                if name not in workbook:
                    raise ValueError(f"Missing tracker worksheet: {name}")
            prior = _registry(workbook, data["evidence_id"])
            if prior is not None:
                # Persist a registry bootstrap only when creating a destination
                # from the template; never rewrite an existing duplicate import.
                if not workbook_path.exists():
                    _refresh_formulas(workbook)
                    _atomic_save(workbook, workbook_path)
                return {"run_id": prior, "imported": False, "workbook_path": str(workbook_path), "evidence_id": data["evidence_id"]}
            limits = _bounds(workbook["Run Log"], "RunConditions")
            available = [str(workbook["Run Log"].cell(row, 1).value) for row in range(limits.min_row + 1, limits.max_row + 1)]
            run_id = next((candidate for candidate in available if _slot_is_empty(workbook, candidate)), None)
            if run_id is None:
                run_id = _extend_slot(workbook)
            _fill_run(workbook, run_id, run_dir, data, metadata or {})
            _refresh_formulas(workbook)
            workbook["Results"]["B4"] = run_id
            _atomic_save(workbook, workbook_path)
            return {"run_id": run_id, "imported": True, "workbook_path": str(workbook_path), "evidence_id": data["evidence_id"]}
        finally:
            workbook.close()
