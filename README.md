# Same-model CPU inference with scikit-learn and ONNX Runtime

This COMP 4330 project evaluates probability inference for one trained Random
Forest through scikit-learn and ONNX Runtime, then measures the effect of disabling
ONNX ZipMap. It studies execution and output representation; it does not train
different models to compare classification accuracy.

## Automatic experiment and Excel recording

### Start from GitHub

1. Install **64-bit Python 3.12**, preferably **3.12.10**, with the Python launcher.
2. Clone this repository to a local folder. Open the folder containing this README,
   `Setup_Environment.cmd`, and `Run_Experiments.cmd`.
3. Double-click **Setup_Environment.cmd** once. It creates a local `.venv`, installs
   the packages in `requirements-tracker.txt`, and checks the environment. Internet
   access is required for this installation. Wait for `Setup completed`.
4. Close **ONNX_Experiment_Tracker.xlsx** in Excel. Plug in the computer, keep a fixed
   Windows power mode, and close heavy background workloads.
5. Double-click **Run_Experiments.cmd**. Enter your name or student ID, plugged-in
   status, actual power mode, and background workload.
6. After the launcher completes, open **ONNX_Experiment_Tracker.xlsx** to review the
   automatically recorded runs. Retain the corresponding `output/runs/` and `logs/`
   folders with the workbook when sharing results with the team.

The launcher performs **four full experiments sequentially**: one independent
reproduction followed by three controlled repeats on the same computer. It pauses
for 60 seconds between runs. Keep the first reproduction separate from the three
stability repeats when comparing results.

Each run follows this sequence:

```text
Record conditions -> Run experiment -> Freshly verify -> Import next Excel RUN
```

The next unused RUN is assigned automatically. Existing measurements and notes are
preserved, repeated imports of the same evidence are skipped, and the workbook's
tables expand when the prepared slots are full. A failed experiment or verification
stops the workflow and retains its logs without adding a successful Excel record.

The recorder uses Python and does not require Microsoft Excel or Codex to write
results. Open the workbook in Excel to recalculate and view its formulas. The
recording modules are separate from the measured `onnx_benchmark` sources, preserving
the original experiment's source hashes. Power mode, CPU temperature and background
load are not controlled by the program; maintain and report the actual conditions.

### Run one experiment

From PowerShell in the project root, run:

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -u -m experiment_tools run --operator "Member2" --plugged-in Yes --power-mode "Best performance" --background-load "Idle; other apps closed" --purpose "Independent reproduction"
```

Replace the example metadata with your actual conditions. This command performs
one full experiment, verifies it and imports it into the root workbook. No virtual
environment activation is needed. Each invocation creates a unique output folder.

To use the interactive launcher for just one experiment:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_team_experiments.ps1 -RunCount 1
```

### Find the records and logs

| Location | Contents |
| --- | --- |
| `ONNX_Experiment_Tracker.xlsx` | Results, Run Log, run settings, Correctness and an English Guide |
| `output/runs/<run-id>/` | Models, frozen split, raw timings, summary, figures, environment, provenance and verification |
| `output/runs/<run-id>/run_conditions.json` | Operator, purpose, power conditions, background workload and workflow references |
| `logs/<workflow-id>/console.log` | Benchmark and verification console output |
| `logs/<workflow-id>/workflow.json` | Phase status, commands, exit codes, timestamps and Excel import result |
| `logs/<workflow-id>/events.jsonl` | Timestamped workflow events |
| `logs/setup_*.log` | Environment installation transcript |
| `logs/team_*.log` | Transcript for a sequential team session |

The historical experiment remains `RUN-001`. `RUN-002` records the full local
automation validation and is labelled `Workflow validation`; exclude it from the
teammate's three-repeat comparison. A fresh clone containing this workbook starts
recording at `RUN-003`; an existing workbook continues at its next available RUN.

### Retry an import or combine teammate results

If an experiment passed verification but Excel was open or the workbook could not
be updated, close Excel and import the completed directory:

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -u -m experiment_tools import --output "output\runs\THE_COMPLETED_RUN_DIRECTORY"
```

Use the actual path printed in the workflow log. Import runs a fresh verification
and then updates Excel; it does not retrain or remeasure. Saved conditions are
reused, and importing evidence that is already recorded does not create another RUN.
If replacement is denied, the error identifies a recoverable workbook file.

To combine a teammate's evidence, keep their complete result directory and import
it into one master workbook instead of replacing its worksheets:

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -u -m experiment_tools import --output "C:\ReturnedEvidence\Member2\THE_RUN_DIRECTORY" --workbook ".\ONNX_Experiment_Tracker.xlsx"
```

The master assigns its own next RUN and keeps the operator, conditions and evidence
identity. Retain the source directory after importing. Use the same benchmark source
version that produced the evidence, because verification checks source hashes.

## Project structure

The main source files and runtime directories are:

```text
scikit-learn_ONNX/
|-- README.md                          # GitHub setup, operation and interpretation
|-- .gitignore                         # Local environment/cache exclusions
|-- Setup_Environment.cmd               # First-time Windows setup entry point
|-- Run_Experiments.cmd                 # Interactive four-run entry point
|-- ONNX_Experiment_Tracker.xlsx        # Working record; updated after verification
|-- requirements.txt                   # Compatible benchmark runtime versions
|-- requirements-lock.txt              # Exact original benchmark/development versions
|-- requirements-tracker.txt           # Pinned runtime plus Excel recording packages
|-- requirements-dev.txt               # Development dependencies
|-- requirements-docs.txt              # Optional document-building dependencies
|-- pyproject.toml                     # Type-check configuration
|-- onnx_benchmark/                    # Measured experiment implementation
|   |-- __main__.py                    # Native thread limits and CLI entry point
|   |-- cli.py                         # run / report / verify orchestration
|   |-- config.py                      # Fixed experiment defaults
|   |-- data_model.py                  # Dataset split, forest and timing batches
|   |-- conversion.py                  # Two ONNX output representations
|   |-- runtime.py                     # sklearn and ONNX Runtime adapters
|   |-- validation.py                  # Label and probability checks
|   |-- benchmark.py                   # Warm-up, timing and summary statistics
|   |-- reporting.py                   # Result document and figures
|   |-- environment.py                # Machine, package and thread metadata
|   `-- artifacts.py                  # JSON and provenance helpers
|-- experiment_tools/                  # Automatic recording workflow
|   |-- __main__.py                    # python -m experiment_tools
|   |-- workflow.py                    # Run, log, freshly verify and import
|   `-- excel_tracker.py               # Next RUN, duplicate detection and safe saves
|-- scripts/
|   |-- setup_environment.ps1          # Install and validate the local environment
|   `-- run_team_experiments.ps1        # Prompt once and execute sequential runs
|-- templates/
|   `-- ONNX_Experiment_Tracker.xlsx    # Used when the target workbook is absent
|-- tests/
|   |-- test_pipeline.py               # Real conversion/inference and evidence checks
|   |-- test_workflow.py               # Phase order, failures, recovery and logging
|   `-- test_excel_tracker.py          # Import, duplicates, growth and safe writing
|-- output/
|   |-- experiment/                    # Original saved benchmark evidence
|   `-- runs/                          # New automatically generated experiments
|-- logs/                              # Setup, session and per-workflow logs
`-- docs/                              # Optional local coursework notes and drafts
```

`output/runs/` and `logs/` are populated as experiments are performed. Each user
creates their own `.venv`; it and temporary/cache folders are ignored by Git.
`docs/` is also currently excluded by `.gitignore`, so the instructions needed to
run and recover experiments are included directly in this README.

## Run the benchmark without Excel recording

Use 64-bit Python 3.12. Run the following from this project directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m onnx_benchmark run --output output/reproduction
.\.venv\Scripts\python.exe -m onnx_benchmark verify --output output/reproduction
```

The lock file contains the exact packages installed for the delivered local run,
including mypy for development. `requirements.txt` describes compatible runtime
ranges; prefer the lock file when reproducing the saved results. No dataset
download, network service, GPU, or web server is needed once packages are installed.

For stable timing, use an idle machine, keep it plugged in, choose a consistent
power mode, and close other compute workloads. Record those conditions yourself;
the program does not control CPU placement, turbo behavior or OS power policy.
Timing magnitudes can vary even with the same machine and package versions.

The default run saves to `output/experiment`. Existing non-empty output directories
are rejected to preserve evidence. Use a different output path for each new run.
Figure regeneration and verification do not retrain or remeasure:

```powershell
.\.venv\Scripts\python.exe -m onnx_benchmark report --output output/experiment
.\.venv\Scripts\python.exe -m onnx_benchmark verify --output output/experiment
```

Verification writes fresh checks under `verification/`, reloads the saved models,
checks source/model hashes and split isolation, and recomputes statistics from raw
timings. If inference source files change, start a new run; an old run is evidence
for its recorded code, not for a modified implementation.

## Experimental design

### Tree-count study and profiling

The separate study compares **10, 100 and 500 trees**, with three independent
repeats of each forest size. It keeps the existing data split, seed 42,
single-thread CPU settings, four batch sizes, and three runtime variants.
Each child experiment trains one forest and converts that same forest twice.
The execution order rotates across repeats: `10/100/500`, `100/500/10`, then
`500/10/100`. With the full defaults, the study records **108,000 timed calls**.

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -u -m onnx_benchmark study --output output/tree_study_20261104 --trees 10 100 500 --repeats 3 --profile-calls 100
.\.venv\Scripts\python.exe -B -X utf8 -m onnx_benchmark study-verify --output output/tree_study_20261104
```

`study` defaults to `output/tree_study`. Use a new empty directory for every study.
The options `--warmup 50 --rounds 10 --calls 100` retain the full measurement
protocol; smaller values are for development checks. All steady-state timings
finish before profiling or plotting begins. Profiling uses the saved models and
inputs in separate passes, so profiler overhead is excluded from speedup results.

Study outputs include `results_zh.md` (Chinese analysis), `runs.csv` (individual
run summaries), `summary.csv` (cross-repeat statistics), `model_sizes.csv`,
`setup_times.csv`, `profiling_summary.csv`, and four PNG figures. Each child
directory retains its models, data, timings, correctness checks, environment,
and provenance, plus raw cProfile `.pstats` and ONNX Runtime profiling JSON.
Serialized file sizes, tree node counts, and maximum depth describe model size;
file size is not a measurement of runtime memory consumption.

Speedup is calculated within each independent run using the corresponding
scikit-learn baseline, then summarized as the median and observed minimum/maximum
across repeats. Calls within one run are not treated as independent experiments.
Preparation durations remain separate from steady-state inference. cProfile
reports function self/cumulative times; its category summaries sum self time
only. ORT summaries report actual operator events within measured `model_run`
windows, excluding initialization and warmup.

The two profilers observe different execution boundaries and add overhead.
Their numbers are not a common, exact breakdown of benchmark latency.
scikit-learn already uses compiled tree traversal; the analysis examines input
validation, Python tree dispatch, probability aggregation, native ensemble
execution, and output representation. The ZipMap on/off latency difference
includes the whole graph and returned-object path, rather than isolating one
operator. Larger models are not assumed to have a greater speedup.

To regenerate the study report and figures without training or timing again:

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m onnx_benchmark study-report --output output/tree_study_20261104
```

`study-verify` reloads child models, checks saved evidence/source hashes and
recomputes the combined tables. Changing benchmark sources makes old runs
historical evidence for their recorded version; verification remains strict.
The Excel launcher above continues to run the original single-model workflow;
study results are saved separately and are not automatically imported into its
100-tree tracker.

### Original single-model experiment

| Setting | Fixed choice |
| --- | --- |
| Dataset | `sklearn.datasets.load_breast_cancer`: 569 samples, 30 features, 2 classes |
| Split | 80/20 stratified, seed 42; 455 training and 114 distinct test samples |
| Model | RandomForestClassifier, 100 trees, seed 42, `n_jobs=1`; other parameters recorded |
| Representation | Train and infer with contiguous float32 inputs |
| ONNX conversion | Requested ceilings: standard opset 17, ai.onnx.ml 3; actual imports recorded per graph |
| Variants | scikit-learn `predict_proba`, ORT ZipMap on, ORT ZipMap off |
| Batches | 1, 16, 64, 256 samples per call |
| Execution | CPU only; ORT sequential, intra/inter-op threads 1, graph optimizations enabled |
| Native libraries | threadpoolctl limits plus native thread environment variables set to 1 |
| Timing | 50 warmups; 10 rounds of 100 calls per variant per batch, rotating variant order |

Only one forest is trained. Both ONNX graphs are converted from that forest, and
the original saved forest is reloaded for the baseline. Fixed shuffled test inputs
are repeated when a batch exceeds the test set. These are artificial timing loads,
not additional independent data. Accuracy uses only the 114 original test cases.

The converter may emit older operator versions within the requested ceilings.
For the delivered run, ZipMap-on imports standard opset 9 / ai.onnx.ml 1;
ZipMap-off imports standard opset 17 / ai.onnx.ml 1. Its repeated standard import
entries are retained in the manifest. Both original converted graphs passed the
ONNX checker and executed in ORT. Requested targets should not be reported as
the actual graph versions.

Each timer covers a probability API call and the construction of its returned
objects. ORT fetches only its probability output; label inference is validated
separately. ZipMap-on materializes a sequence of dictionaries, while ZipMap-off
returns a probability tensor. Python dictionary-to-matrix normalization happens
outside timing. Input preparation, CSV output, result destruction, training,
deserialization, and session creation are excluded. Setup durations are recorded
once, separately, and do not represent cold-start distributions.

This measures the practical cost of these APIs, including scikit-learn's Python
dispatch and output handling. It is not a measurement of pure tree computation or
the isolated ZipMap operator. A downstream consumer needing another representation
may incur additional costs beyond this experiment.

## Correctness and metrics

Before benchmarking, the program checks labels and probabilities on the original
test set and all performance batches. It requires 100% agreement with scikit-learn,
finite probabilities within [0, 1], rows summing to one, and maximum absolute
probability difference at most `1e-5`. Failure saves the diagnostic JSON and stops
before timing.

Median latency, P25, P75 and P95 summarize the 1,000 call timings in each of the
12 configurations. `throughput = batch_size / median_seconds`;
`speedup = sklearn_median / compared_median`. Timing quantiles describe this
observed run; they are not confidence intervals across machines or datasets.

## Outputs and code organization

The run directory includes:

- `results.md`, `summary.csv`, and `figures/`: actual results and three charts.
- `timings.csv`: 12,000 individual timed calls with round, order and call index.
- `correctness.json`: held-out accuracy, agreement, probability errors and batch checks.
- `dataset.npz`, `dataset.json`, `batch_indices.npz`: frozen data, split and synthetic load indices.
- `random_forest.joblib`, `onnx_zipmap.onnx`, `onnx_tensor.onnx`: the same-model artifacts.
- `config.json`, `model_parameters.json`, `conversion.json`, `runtime_settings.json`:
  precise experimental choices and graph outputs.
- `environment.json`, `setup_times.json`, `provenance.json`, `status.json`:
  machine/packages, setup observations, source/artifact hashes and completion state.
- `verification/`: checks from reloading a saved run and recalculating its summary.

`onnx_benchmark` separates configuration, shared artifact helpers, data/model,
conversion, runtime adapters, prediction validation, timing, environment reporting,
plotting and orchestration. `python -m onnx_benchmark` is the entry point.
The `run`, `report` and `verify` commands accept `--output`. `run` also accepts
`--warmup`, `--rounds` and `--calls` for development checks. Reduced runs should
not be substituted for the delivered full experiment.

## Development validation

```powershell
.\.venv\Scripts\python.exe -m mypy
.\.venv\Scripts\python.exe -m mypy experiment_tools scripts/package_team_runner.py
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

The tests exercise real conversion/inference, repeatable splits and loads,
prediction validation, original-output preservation and consistency between raw
timing and summary data. They are separate from the delivered full benchmark.

## Coursework documents and next steps

`output/docx/project_proposal.docx` is the editable English Word proposal, with
native text and tables. `output/pdf/project_proposal.pdf` is the earlier PDF draft.
These are local coursework artifacts rather than required runner files. Replace
the team identity blanks and Member 1-5 labels with
real names/IDs before submission. Tutor acceptance of the topic does not determine
its difficulty coefficient or eventual score.

The delivered PDF was generated with ReportLab 4.4.9 and visually checked on all
four pages. The editable DOCX was generated with python-docx 1.2.0, rendered with
Microsoft Word and visually checked on all four pages; its text and four tables
remain native editable Word content. Changes to either document should be followed
by another layout check. Document builders in `scripts/` are optional and require
their Markdown source files and `requirements-docs.txt` dependencies.

`docs/project_plan.md` records responsibility and submission milestones. Review
the actual results, prepare the six-minute presentation and Q&A, and complete the
English final report within the official 2,000-word limit. The first local results
do not establish general speedup for other models, machines or workloads.

`docs/source_analysis.md` traces the actual installed and version-pinned upstream
implementation paths. `docs/final_report_draft.md` and
`docs/presentation_outline.md` provide an English report draft and a six-minute
presentation outline based on the saved results; review them with your team.

The GitHub repository is the project sharing location. Keep each experiment's full
evidence directory and logs alongside its workbook records. Member details, the
formatted final report, the actual slide deck, rehearsal and the official OLE
submission still need to be completed by the team.

Official references:

- scikit-learn RandomForestClassifier:
  https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- Breast Cancer dataset:
  https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
- sklearn-onnx ZipMap comparison:
  https://onnx.ai/sklearn-onnx/auto_examples/plot_convert_zipmap.html
- ONNX Runtime thread controls:
  https://onnxruntime.ai/docs/performance/tune-performance/threading.html
