# CPU inference benchmark results

This run compares the same trained 100-tree Random Forest on the Breast Cancer Wisconsin dataset using scikit-learn and two ONNX Runtime output representations. All measurements come from this run on one CPU machine. Hardware and software versions are recorded in [environment.json](environment.json); the data split is recorded in [dataset.json](dataset.json), model parameters in [model_parameters.json](model_parameters.json), effective runtime and thread settings in [runtime_settings.json](runtime_settings.json), and experiment and timing settings in [config.json](config.json).

## All measured configurations

| Backend | Batch size | Timed calls | Median (ms) | P25-P75 (ms) | P95 (ms) | Throughput (samples/s) | Speedup vs scikit-learn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scikit-learn | 1 | 1000 | 2.56825 | 2.5503-2.59322 | 2.65225 | 389.37 | 1.0000x |
| scikit-learn | 16 | 1000 | 2.6286 | 2.60878-2.6626 | 2.77011 | 6086.89 | 1.0000x |
| scikit-learn | 64 | 1000 | 2.76105 | 2.7414-2.79698 | 2.88981 | 23179.6 | 1.0000x |
| scikit-learn | 256 | 1000 | 3.0321 | 3.01607-3.06022 | 3.12231 | 84429.9 | 1.0000x |
| ONNX Runtime (ZipMap on) | 1 | 1000 | 0.0077 | 0.0076-0.0079 | 0.0084 | 129870 | 333.5390x |
| ONNX Runtime (ZipMap on) | 16 | 1000 | 0.0239 | 0.0235-0.0253 | 0.0271 | 669456 | 109.9833x |
| ONNX Runtime (ZipMap on) | 64 | 1000 | 0.1238 | 0.1216-0.127125 | 0.131405 | 516963 | 22.3025x |
| ONNX Runtime (ZipMap on) | 256 | 1000 | 0.4857 | 0.4816-0.4908 | 0.4995 | 527074 | 6.2427x |
| ONNX Runtime (ZipMap off) | 1 | 1000 | 0.0065 | 0.0064-0.0066 | 0.007 | 153846 | 395.1154x |
| ONNX Runtime (ZipMap off) | 16 | 1000 | 0.0199 | 0.0195-0.0205 | 0.0221 | 804020 | 132.0905x |
| ONNX Runtime (ZipMap off) | 64 | 1000 | 0.1113 | 0.109-0.1146 | 0.119 | 575022 | 24.8073x |
| ONNX Runtime (ZipMap off) | 256 | 1000 | 0.4375 | 0.4335-0.442025 | 0.4489 | 585143 | 6.9305x |

The median is the middle observed per-call timing. P25-P75 shows the central 50% of timings (the interquartile range); P95 is the 95th percentile. These describe timing variation and are not confidence intervals for a population of machines. Throughput is batch size divided by median call time. Speedup is the scikit-learn median divided by the compared backend median for the same batch size: above 1 means faster and below 1 means slower.

![Median latency and interquartile range](figures/latency.png)

![Throughput from median latency](figures/throughput.png)

## ZipMap comparison

| Batch size | Median latency: ZipMap on / ZipMap off |
| ---: | ---: |
| 1 | 1.1846x |
| 16 | 1.2010x |
| 64 | 1.1123x |
| 256 | 1.1102x |

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
