# CPU inference benchmark results

This run compares the same trained 10-tree Random Forest on the Breast Cancer Wisconsin dataset using scikit-learn and two ONNX Runtime output representations. All measurements come from this run on one CPU machine. Hardware and software versions are recorded in [environment.json](environment.json); the data split is recorded in [dataset.json](dataset.json), model parameters in [model_parameters.json](model_parameters.json), effective runtime and thread settings in [runtime_settings.json](runtime_settings.json), and experiment and timing settings in [config.json](config.json).

## All measured configurations

| Backend | Batch size | Timed calls | Median (ms) | P25-P75 (ms) | P95 (ms) | Throughput (samples/s) | Speedup vs scikit-learn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scikit-learn | 1 | 1000 | 0.6134 | 0.5087-0.7201 | 0.93984 | 1630.26 | 1.0000x |
| scikit-learn | 16 | 1000 | 0.64225 | 0.4859-0.743625 | 0.994945 | 24912.4 | 1.0000x |
| scikit-learn | 64 | 1000 | 0.6187 | 0.516025-0.695275 | 0.904935 | 103443 | 1.0000x |
| scikit-learn | 256 | 1000 | 0.73 | 0.6189-0.869525 | 1.34977 | 350685 | 1.0000x |
| ONNX Runtime (ZipMap on) | 1 | 1000 | 0.0161 | 0.0132-0.0177 | 0.02171 | 62111.8 | 38.0994x |
| ONNX Runtime (ZipMap on) | 16 | 1000 | 0.0189 | 0.0182-0.0252 | 0.0282 | 846561 | 33.9815x |
| ONNX Runtime (ZipMap on) | 64 | 1000 | 0.0442 | 0.0324-0.046225 | 0.053705 | 1.44796e+06 | 13.9977x |
| ONNX Runtime (ZipMap on) | 256 | 1000 | 0.14265 | 0.1043-0.1577 | 0.2268 | 1.7946e+06 | 5.1174x |
| ONNX Runtime (ZipMap off) | 1 | 1000 | 0.0143 | 0.0125-0.0153 | 0.0168 | 69930.1 | 42.8951x |
| ONNX Runtime (ZipMap off) | 16 | 1000 | 0.0139 | 0.013675-0.018825 | 0.021 | 1.15108e+06 | 46.2050x |
| ONNX Runtime (ZipMap off) | 64 | 1000 | 0.0259 | 0.0189-0.0268 | 0.0297 | 2.47104e+06 | 23.8880x |
| ONNX Runtime (ZipMap off) | 256 | 1000 | 0.06085 | 0.0424-0.0658 | 0.07679 | 4.20707e+06 | 11.9967x |

The median is the middle observed per-call timing. P25-P75 shows the central 50% of timings (the interquartile range); P95 is the 95th percentile. These describe timing variation and are not confidence intervals for a population of machines. Throughput is batch size divided by median call time. Speedup is the scikit-learn median divided by the compared backend median for the same batch size: above 1 means faster and below 1 means slower.

![Median latency and interquartile range](figures/latency.png)

![Throughput from median latency](figures/throughput.png)

## ZipMap comparison

| Batch size | Median latency: ZipMap on / ZipMap off |
| ---: | ---: |
| 1 | 1.1259x |
| 16 | 1.3597x |
| 64 | 1.7066x |
| 256 | 2.3443x |

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
