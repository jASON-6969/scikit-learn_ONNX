# CPU inference benchmark results

This run compares the same trained 100-tree Random Forest on the Breast Cancer Wisconsin dataset using scikit-learn and two ONNX Runtime output representations. All measurements come from this run on one CPU machine. Hardware and software versions are recorded in [environment.json](environment.json); the data split is recorded in [dataset.json](dataset.json), model parameters in [model_parameters.json](model_parameters.json), effective runtime and thread settings in [runtime_settings.json](runtime_settings.json), and experiment and timing settings in [config.json](config.json).

## All measured configurations

| Backend | Batch size | Timed calls | Median (ms) | P25-P75 (ms) | P95 (ms) | Throughput (samples/s) | Speedup vs scikit-learn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scikit-learn | 1 | 1000 | 2.5571 | 2.5402-2.58383 | 2.73554 | 391.068 | 1.0000x |
| scikit-learn | 16 | 1000 | 2.6075 | 2.5917-2.62893 | 2.70881 | 6136.15 | 1.0000x |
| scikit-learn | 64 | 1000 | 2.7459 | 2.73128-2.77393 | 2.88048 | 23307.5 | 1.0000x |
| scikit-learn | 256 | 1000 | 3.03575 | 3.0183-3.06155 | 3.13236 | 84328.4 | 1.0000x |
| ONNX Runtime (ZipMap on) | 1 | 1000 | 0.0077 | 0.0076-0.0079 | 0.0083 | 129870 | 332.0909x |
| ONNX Runtime (ZipMap on) | 16 | 1000 | 0.0237 | 0.0233-0.0242 | 0.0271 | 675105 | 110.0211x |
| ONNX Runtime (ZipMap on) | 64 | 1000 | 0.1263 | 0.123-0.1292 | 0.1355 | 506730 | 21.7411x |
| ONNX Runtime (ZipMap on) | 256 | 1000 | 0.4862 | 0.4822-0.4905 | 0.5096 | 526532 | 6.2438x |
| ONNX Runtime (ZipMap off) | 1 | 1000 | 0.0065 | 0.0064-0.0066 | 0.0069 | 153846 | 393.4000x |
| ONNX Runtime (ZipMap off) | 16 | 1000 | 0.0196 | 0.0193-0.0202 | 0.022405 | 816327 | 133.0357x |
| ONNX Runtime (ZipMap off) | 64 | 1000 | 0.1128 | 0.1098-0.1157 | 0.121 | 567376 | 24.3431x |
| ONNX Runtime (ZipMap off) | 256 | 1000 | 0.4371 | 0.4333-0.4417 | 0.466705 | 585678 | 6.9452x |

The median is the middle observed per-call timing. P25-P75 shows the central 50% of timings (the interquartile range); P95 is the 95th percentile. These describe timing variation and are not confidence intervals for a population of machines. Throughput is batch size divided by median call time. Speedup is the scikit-learn median divided by the compared backend median for the same batch size: above 1 means faster and below 1 means slower.

![Median latency and interquartile range](figures/latency.png)

![Throughput from median latency](figures/throughput.png)

## ZipMap comparison

| Batch size | Median latency: ZipMap on / ZipMap off |
| ---: | ---: |
| 1 | 1.1846x |
| 16 | 1.2092x |
| 64 | 1.1197x |
| 256 | 1.1123x |

A ratio above 1 favors disabling ZipMap; below 1 favors enabling it. This is a measured change between converted graphs and their output representations, not an isolated measurement of one operator's cost.

![ZipMap median latency ratios](figures/zipmap_speedup.png)

## Prediction correctness

Overall check passed: **yes**. Original held-out cases: **114**. Absolute probability tolerance: **1e-05**.

Baseline scikit-learn accuracy: **95.6140%**.

| Backend | Label agreement | Accuracy | Maximum absolute probability error | Finite probabilities | Normalized probability rows |
| --- | ---: | ---: | ---: | --- | --- |
| scikit-learn | 100.0000% | 95.6140% | 0 | yes | yes |
| ONNX Runtime (ZipMap on) | 100.0000% | 95.6140% | 6.55651e-07 | yes | yes |
| ONNX Runtime (ZipMap off) | 100.0000% | 95.6140% | 6.55651e-07 | yes | yes |

## Measurement scope and limitations

- Accuracy and agreement use the original held-out cases. Repeated inputs used to form synthetic performance batches are not additional independent accuracy samples.
- Timings cover warmed, in-process inference calls and the output objects materialized by those APIs. Model loading, session creation, input preparation, output object destruction, and dictionary-to-matrix conversion for correctness checks are outside the timed region.
- scikit-learn is measured through predict_proba; ONNX Runtime fetches the probability output only. ZipMap on returns dictionaries and ZipMap off returns a tensor. This compares probability-producing APIs rather than isolated model computation; scikit-learn's Python tree dispatch is included. Consumers needing a different final format may incur additional conversion costs.
- OS power settings and thread placement are not controlled by this benchmark. Record the machine's power mode, concurrent workload, and other relevant run conditions manually when interpreting or repeating these measurements.
- One machine, one dataset, and one Random Forest configuration cannot establish performance for other models, hardware, GPU execution, or production workloads. Repeated fixed inputs and warmed sessions also differ from cold requests and changing inputs.
- Conclusions should follow the observed ratios and timing spread. A slower ONNX result is still a valid observation; no acceleration factor was assumed in advance.

Raw per-call observations are retained in [timings.csv](timings.csv); the table above is generated from [summary.csv](summary.csv). Prediction checks are recorded in [correctness.json](correctness.json).

Setup durations are recorded separately in [setup_times.json](setup_times.json) and are excluded from the steady-state inference ratios above.
