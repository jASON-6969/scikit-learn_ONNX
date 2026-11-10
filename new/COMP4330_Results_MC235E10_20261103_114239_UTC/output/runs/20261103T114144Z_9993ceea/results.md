# CPU inference benchmark results

This run compares the same trained 100-tree Random Forest on the Breast Cancer Wisconsin dataset using scikit-learn and two ONNX Runtime output representations. All measurements come from this run on one CPU machine. Hardware and software versions are recorded in [environment.json](environment.json); the data split is recorded in [dataset.json](dataset.json), model parameters in [model_parameters.json](model_parameters.json), effective runtime and thread settings in [runtime_settings.json](runtime_settings.json), and experiment and timing settings in [config.json](config.json).

## All measured configurations

| Backend | Batch size | Timed calls | Median (ms) | P25-P75 (ms) | P95 (ms) | Throughput (samples/s) | Speedup vs scikit-learn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scikit-learn | 1 | 1000 | 2.61335 | 2.5872-2.6518 | 2.76387 | 382.651 | 1.0000x |
| scikit-learn | 16 | 1000 | 2.65825 | 2.6333-2.689 | 2.77633 | 6019 | 1.0000x |
| scikit-learn | 64 | 1000 | 2.79545 | 2.76118-2.83003 | 2.94241 | 22894.3 | 1.0000x |
| scikit-learn | 256 | 1000 | 3.07875 | 3.046-3.12605 | 3.26306 | 83150.6 | 1.0000x |
| ONNX Runtime (ZipMap on) | 1 | 1000 | 0.0078 | 0.0076-0.008 | 0.022205 | 128205 | 335.0449x |
| ONNX Runtime (ZipMap on) | 16 | 1000 | 0.0239 | 0.0234-0.0251 | 0.0276 | 669456 | 111.2238x |
| ONNX Runtime (ZipMap on) | 64 | 1000 | 0.1263 | 0.123-0.1297 | 0.13671 | 506730 | 22.1334x |
| ONNX Runtime (ZipMap on) | 256 | 1000 | 0.4882 | 0.4835-0.494525 | 0.502005 | 524375 | 6.3063x |
| ONNX Runtime (ZipMap off) | 1 | 1000 | 0.0065 | 0.0064-0.0067 | 0.007 | 153846 | 402.0538x |
| ONNX Runtime (ZipMap off) | 16 | 1000 | 0.0197 | 0.0194-0.0203 | 0.022805 | 812183 | 134.9365x |
| ONNX Runtime (ZipMap off) | 64 | 1000 | 0.11395 | 0.1106-0.1171 | 0.123105 | 561650 | 24.5323x |
| ONNX Runtime (ZipMap off) | 256 | 1000 | 0.4409 | 0.4368-0.4457 | 0.45391 | 580631 | 6.9829x |

The median is the middle observed per-call timing. P25-P75 shows the central 50% of timings (the interquartile range); P95 is the 95th percentile. These describe timing variation and are not confidence intervals for a population of machines. Throughput is batch size divided by median call time. Speedup is the scikit-learn median divided by the compared backend median for the same batch size: above 1 means faster and below 1 means slower.

![Median latency and interquartile range](figures/latency.png)

![Throughput from median latency](figures/throughput.png)

## ZipMap comparison

| Batch size | Median latency: ZipMap on / ZipMap off |
| ---: | ---: |
| 1 | 1.2000x |
| 16 | 1.2132x |
| 64 | 1.1084x |
| 256 | 1.1073x |

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
