# CPU inference benchmark results

This run compares the same trained 500-tree Random Forest on the Breast Cancer Wisconsin dataset using scikit-learn and two ONNX Runtime output representations. All measurements come from this run on one CPU machine. Hardware and software versions are recorded in [environment.json](environment.json); the data split is recorded in [dataset.json](dataset.json), model parameters in [model_parameters.json](model_parameters.json), effective runtime and thread settings in [runtime_settings.json](runtime_settings.json), and experiment and timing settings in [config.json](config.json).

## All measured configurations

| Backend | Batch size | Timed calls | Median (ms) | P25-P75 (ms) | P95 (ms) | Throughput (samples/s) | Speedup vs scikit-learn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scikit-learn | 1 | 1000 | 20.9809 | 17.8528-22.8657 | 30.077 | 47.6623 | 1.0000x |
| scikit-learn | 16 | 1000 | 21.4904 | 18.1494-23.4924 | 30.4359 | 744.518 | 1.0000x |
| scikit-learn | 64 | 1000 | 21.4604 | 19.0666-23.4741 | 28.7919 | 2982.24 | 1.0000x |
| scikit-learn | 256 | 1000 | 24.8189 | 22.1585-26.6084 | 34.1829 | 10314.7 | 1.0000x |
| ONNX Runtime (ZipMap on) | 1 | 1000 | 0.024 | 0.0168-0.0268 | 0.0316 | 41666.7 | 874.2062x |
| ONNX Runtime (ZipMap on) | 16 | 1000 | 0.14535 | 0.1055-0.1619 | 0.27521 | 110079 | 147.8528x |
| ONNX Runtime (ZipMap on) | 64 | 1000 | 0.94285 | 0.807525-1.0243 | 1.13148 | 67879.3 | 22.7612x |
| ONNX Runtime (ZipMap on) | 256 | 1000 | 3.62925 | 3.15203-3.82492 | 5.65779 | 70538 | 6.8386x |
| ONNX Runtime (ZipMap off) | 1 | 1000 | 0.0188 | 0.015-0.0228 | 0.025 | 53191.5 | 1116.0080x |
| ONNX Runtime (ZipMap off) | 16 | 1000 | 0.1398 | 0.103475-0.159325 | 0.27382 | 114449 | 153.7225x |
| ONNX Runtime (ZipMap off) | 64 | 1000 | 0.9168 | 0.77685-0.991225 | 1.08303 | 69808 | 23.4079x |
| ONNX Runtime (ZipMap off) | 256 | 1000 | 3.5644 | 3.20445-3.75992 | 6.11494 | 71821.3 | 6.9630x |

The median is the middle observed per-call timing. P25-P75 shows the central 50% of timings (the interquartile range); P95 is the 95th percentile. These describe timing variation and are not confidence intervals for a population of machines. Throughput is batch size divided by median call time. Speedup is the scikit-learn median divided by the compared backend median for the same batch size: above 1 means faster and below 1 means slower.

![Median latency and interquartile range](figures/latency.png)

![Throughput from median latency](figures/throughput.png)

## ZipMap comparison

| Batch size | Median latency: ZipMap on / ZipMap off |
| ---: | ---: |
| 1 | 1.2766x |
| 16 | 1.0397x |
| 64 | 1.0284x |
| 256 | 1.0182x |

A ratio above 1 favors disabling ZipMap; below 1 favors enabling it. This is a measured change between converted graphs and their output representations, not an isolated measurement of one operator's cost.

![ZipMap median latency ratios](figures/zipmap_speedup.png)

## Prediction correctness

Overall check passed: **yes**. Original held-out cases: **114**. Absolute probability tolerance: **1e-05**.

Baseline scikit-learn accuracy: **95.6140%**.

| Backend | Label agreement | Accuracy | Maximum absolute probability error | Finite probabilities | Normalized probability rows |
| --- | ---: | ---: | ---: | --- | --- |
| scikit-learn | 100.0000% | 95.6140% | 0 | yes | yes |
| ONNX Runtime (ZipMap on) | 100.0000% | 95.6140% | 5.78165e-06 | yes | yes |
| ONNX Runtime (ZipMap off) | 100.0000% | 95.6140% | 5.78165e-06 | yes | yes |

## Measurement scope and limitations

- Accuracy and agreement use the original held-out cases. Repeated inputs used to form synthetic performance batches are not additional independent accuracy samples.
- Timings cover warmed, in-process inference calls and the output objects materialized by those APIs. Model loading, session creation, input preparation, output object destruction, and dictionary-to-matrix conversion for correctness checks are outside the timed region.
- scikit-learn is measured through predict_proba; ONNX Runtime fetches the probability output only. ZipMap on returns dictionaries and ZipMap off returns a tensor. This compares probability-producing APIs rather than isolated model computation; scikit-learn's Python tree dispatch is included. Consumers needing a different final format may incur additional conversion costs.
- OS power settings and thread placement are not controlled by this benchmark. Record the machine's power mode, concurrent workload, and other relevant run conditions manually when interpreting or repeating these measurements.
- One machine, one dataset, and one Random Forest configuration cannot establish performance for other models, hardware, GPU execution, or production workloads. Repeated fixed inputs and warmed sessions also differ from cold requests and changing inputs.
- Conclusions should follow the observed ratios and timing spread. A slower ONNX result is still a valid observation; no acceleration factor was assumed in advance.

Raw per-call observations are retained in [timings.csv](timings.csv); the table above is generated from [summary.csv](summary.csv). Prediction checks are recorded in [correctness.json](correctness.json).

Setup durations are recorded separately in [setup_times.json](setup_times.json) and are excluded from the steady-state inference ratios above.
