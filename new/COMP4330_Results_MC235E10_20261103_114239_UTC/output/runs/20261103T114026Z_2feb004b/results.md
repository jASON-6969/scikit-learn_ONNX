# CPU inference benchmark results

This run compares the same trained 100-tree Random Forest on the Breast Cancer Wisconsin dataset using scikit-learn and two ONNX Runtime output representations. All measurements come from this run on one CPU machine. Hardware and software versions are recorded in [environment.json](environment.json); the data split is recorded in [dataset.json](dataset.json), model parameters in [model_parameters.json](model_parameters.json), effective runtime and thread settings in [runtime_settings.json](runtime_settings.json), and experiment and timing settings in [config.json](config.json).

## All measured configurations

| Backend | Batch size | Timed calls | Median (ms) | P25-P75 (ms) | P95 (ms) | Throughput (samples/s) | Speedup vs scikit-learn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scikit-learn | 1 | 1000 | 2.6192 | 2.60098-2.65135 | 2.76378 | 381.796 | 1.0000x |
| scikit-learn | 16 | 1000 | 2.63425 | 2.6054-2.66715 | 2.73965 | 6073.84 | 1.0000x |
| scikit-learn | 64 | 1000 | 2.7439 | 2.7275-2.7727 | 2.92341 | 23324.5 | 1.0000x |
| scikit-learn | 256 | 1000 | 3.04255 | 3.0251-3.07465 | 3.18262 | 84139.9 | 1.0000x |
| ONNX Runtime (ZipMap on) | 1 | 1000 | 0.0079 | 0.0078-0.0081 | 0.0085 | 126582 | 331.5443x |
| ONNX Runtime (ZipMap on) | 16 | 1000 | 0.0236 | 0.0232-0.0241 | 0.02811 | 677966 | 111.6208x |
| ONNX Runtime (ZipMap on) | 64 | 1000 | 0.1237 | 0.1215-0.127025 | 0.1309 | 517381 | 22.1819x |
| ONNX Runtime (ZipMap on) | 256 | 1000 | 0.4855 | 0.4815-0.491225 | 0.508305 | 527291 | 6.2668x |
| ONNX Runtime (ZipMap off) | 1 | 1000 | 0.0066 | 0.0065-0.0068 | 0.0072 | 151515 | 396.8485x |
| ONNX Runtime (ZipMap off) | 16 | 1000 | 0.0196 | 0.0192-0.0201 | 0.0213 | 816327 | 134.4005x |
| ONNX Runtime (ZipMap off) | 64 | 1000 | 0.1107 | 0.108675-0.1139 | 0.1173 | 578139 | 24.7868x |
| ONNX Runtime (ZipMap off) | 256 | 1000 | 0.4357 | 0.4326-0.4398 | 0.4478 | 587560 | 6.9831x |

The median is the middle observed per-call timing. P25-P75 shows the central 50% of timings (the interquartile range); P95 is the 95th percentile. These describe timing variation and are not confidence intervals for a population of machines. Throughput is batch size divided by median call time. Speedup is the scikit-learn median divided by the compared backend median for the same batch size: above 1 means faster and below 1 means slower.

![Median latency and interquartile range](figures/latency.png)

![Throughput from median latency](figures/throughput.png)

## ZipMap comparison

| Batch size | Median latency: ZipMap on / ZipMap off |
| ---: | ---: |
| 1 | 1.1970x |
| 16 | 1.2041x |
| 64 | 1.1174x |
| 256 | 1.1143x |

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
