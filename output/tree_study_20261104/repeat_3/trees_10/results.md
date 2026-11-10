# CPU inference benchmark results

This run compares the same trained 10-tree Random Forest on the Breast Cancer Wisconsin dataset using scikit-learn and two ONNX Runtime output representations. All measurements come from this run on one CPU machine. Hardware and software versions are recorded in [environment.json](environment.json); the data split is recorded in [dataset.json](dataset.json), model parameters in [model_parameters.json](model_parameters.json), effective runtime and thread settings in [runtime_settings.json](runtime_settings.json), and experiment and timing settings in [config.json](config.json).

## All measured configurations

| Backend | Batch size | Timed calls | Median (ms) | P25-P75 (ms) | P95 (ms) | Throughput (samples/s) | Speedup vs scikit-learn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scikit-learn | 1 | 1000 | 0.605 | 0.5316-0.68215 | 0.851055 | 1652.89 | 1.0000x |
| scikit-learn | 16 | 1000 | 0.6142 | 0.48205-0.723725 | 0.92038 | 26050.1 | 1.0000x |
| scikit-learn | 64 | 1000 | 0.60905 | 0.462725-0.674175 | 0.887845 | 105082 | 1.0000x |
| scikit-learn | 256 | 1000 | 0.64555 | 0.48105-0.748475 | 1.03002 | 396561 | 1.0000x |
| ONNX Runtime (ZipMap on) | 1 | 1000 | 0.013 | 0.0098-0.0136 | 0.015 | 76923.1 | 46.5385x |
| ONNX Runtime (ZipMap on) | 16 | 1000 | 0.02 | 0.0146-0.022 | 0.05352 | 800000 | 30.7100x |
| ONNX Runtime (ZipMap on) | 64 | 1000 | 0.042 | 0.0316-0.0458 | 0.05342 | 1.52381e+06 | 14.5012x |
| ONNX Runtime (ZipMap on) | 256 | 1000 | 0.1274 | 0.0901-0.1437 | 0.157315 | 2.00942e+06 | 5.0671x |
| ONNX Runtime (ZipMap off) | 1 | 1000 | 0.0111 | 0.0084-0.012 | 0.013305 | 90090.1 | 54.5045x |
| ONNX Runtime (ZipMap off) | 16 | 1000 | 0.0152 | 0.0144-0.0159 | 0.085115 | 1.05263e+06 | 40.4079x |
| ONNX Runtime (ZipMap off) | 64 | 1000 | 0.0225 | 0.0166-0.0238 | 0.0257 | 2.84444e+06 | 27.0689x |
| ONNX Runtime (ZipMap off) | 256 | 1000 | 0.0469 | 0.0366-0.0604 | 0.07953 | 5.45842e+06 | 13.7644x |

The median is the middle observed per-call timing. P25-P75 shows the central 50% of timings (the interquartile range); P95 is the 95th percentile. These describe timing variation and are not confidence intervals for a population of machines. Throughput is batch size divided by median call time. Speedup is the scikit-learn median divided by the compared backend median for the same batch size: above 1 means faster and below 1 means slower.

![Median latency and interquartile range](figures/latency.png)

![Throughput from median latency](figures/throughput.png)

## ZipMap comparison

| Batch size | Median latency: ZipMap on / ZipMap off |
| ---: | ---: |
| 1 | 1.1712x |
| 16 | 1.3158x |
| 64 | 1.8667x |
| 256 | 2.7164x |

A ratio above 1 favors disabling ZipMap; below 1 favors enabling it. This is a measured change between converted graphs and their output representations, not an isolated measurement of one operator's cost.

![ZipMap median latency ratios](figures/zipmap_speedup.png)

## Prediction correctness

Overall check passed: **yes**. Original held-out cases: **114**. Absolute probability tolerance: **1e-05**.

Baseline scikit-learn accuracy: **93.8596%**.

| Backend | Label agreement | Accuracy | Maximum absolute probability error | Finite probabilities | Normalized probability rows |
| --- | ---: | ---: | ---: | --- | --- |
| scikit-learn | 100.0000% | 93.8596% | 0 | yes | yes |
| ONNX Runtime (ZipMap on) | 100.0000% | 93.8596% | 1.19209e-07 | yes | yes |
| ONNX Runtime (ZipMap off) | 100.0000% | 93.8596% | 1.19209e-07 | yes | yes |

## Measurement scope and limitations

- Accuracy and agreement use the original held-out cases. Repeated inputs used to form synthetic performance batches are not additional independent accuracy samples.
- Timings cover warmed, in-process inference calls and the output objects materialized by those APIs. Model loading, session creation, input preparation, output object destruction, and dictionary-to-matrix conversion for correctness checks are outside the timed region.
- scikit-learn is measured through predict_proba; ONNX Runtime fetches the probability output only. ZipMap on returns dictionaries and ZipMap off returns a tensor. This compares probability-producing APIs rather than isolated model computation; scikit-learn's Python tree dispatch is included. Consumers needing a different final format may incur additional conversion costs.
- OS power settings and thread placement are not controlled by this benchmark. Record the machine's power mode, concurrent workload, and other relevant run conditions manually when interpreting or repeating these measurements.
- One machine, one dataset, and one Random Forest configuration cannot establish performance for other models, hardware, GPU execution, or production workloads. Repeated fixed inputs and warmed sessions also differ from cold requests and changing inputs.
- Conclusions should follow the observed ratios and timing spread. A slower ONNX result is still a valid observation; no acceleration factor was assumed in advance.

Raw per-call observations are retained in [timings.csv](timings.csv); the table above is generated from [summary.csv](summary.csv). Prediction checks are recorded in [correctness.json](correctness.json).

Setup durations are recorded separately in [setup_times.json](setup_times.json) and are excluded from the steady-state inference ratios above.
