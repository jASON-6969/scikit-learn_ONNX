# CPU inference benchmark results

This run compares the same trained 100-tree Random Forest on the Breast Cancer Wisconsin dataset using scikit-learn and two ONNX Runtime output representations. All measurements come from this run on one CPU machine. Hardware and software versions are recorded in [environment.json](environment.json); the data split is recorded in [dataset.json](dataset.json), model parameters in [model_parameters.json](model_parameters.json), effective runtime and thread settings in [runtime_settings.json](runtime_settings.json), and experiment and timing settings in [config.json](config.json).

## All measured configurations

| Backend | Batch size | Timed calls | Median (ms) | P25-P75 (ms) | P95 (ms) | Throughput (samples/s) | Speedup vs scikit-learn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scikit-learn | 1 | 1000 | 4.53 | 3.68825-5.00235 | 7.55619 | 220.751 | 1.0000x |
| scikit-learn | 16 | 1000 | 4.5052 | 3.61718-5.08003 | 6.78021 | 3551.45 | 1.0000x |
| scikit-learn | 64 | 1000 | 4.7093 | 3.89387-5.10872 | 6.53692 | 13590.1 | 1.0000x |
| scikit-learn | 256 | 1000 | 5.2026 | 4.2505-5.69615 | 7.5787 | 49206.2 | 1.0000x |
| ONNX Runtime (ZipMap on) | 1 | 1000 | 0.0146 | 0.0111-0.016 | 0.036155 | 68493.2 | 310.2740x |
| ONNX Runtime (ZipMap on) | 16 | 1000 | 0.04125 | 0.0285-0.0459 | 0.052705 | 387879 | 109.2170x |
| ONNX Runtime (ZipMap on) | 64 | 1000 | 0.13545 | 0.091975-0.15275 | 0.28414 | 472499 | 34.7678x |
| ONNX Runtime (ZipMap on) | 256 | 1000 | 0.61545 | 0.479825-0.693525 | 1.03719 | 415956 | 8.4533x |
| ONNX Runtime (ZipMap off) | 1 | 1000 | 0.0099 | 0.0095-0.014 | 0.026315 | 101010 | 457.5758x |
| ONNX Runtime (ZipMap off) | 16 | 1000 | 0.0352 | 0.0246-0.0395 | 0.078405 | 454545 | 127.9886x |
| ONNX Runtime (ZipMap off) | 64 | 1000 | 0.1127 | 0.1094-0.1205 | 0.16355 | 567879 | 41.7862x |
| ONNX Runtime (ZipMap off) | 256 | 1000 | 0.4792 | 0.3914-0.546 | 0.64776 | 534224 | 10.8568x |

The median is the middle observed per-call timing. P25-P75 shows the central 50% of timings (the interquartile range); P95 is the 95th percentile. These describe timing variation and are not confidence intervals for a population of machines. Throughput is batch size divided by median call time. Speedup is the scikit-learn median divided by the compared backend median for the same batch size: above 1 means faster and below 1 means slower.

![Median latency and interquartile range](figures/latency.png)

![Throughput from median latency](figures/throughput.png)

## ZipMap comparison

| Batch size | Median latency: ZipMap on / ZipMap off |
| ---: | ---: |
| 1 | 1.4747x |
| 16 | 1.1719x |
| 64 | 1.2019x |
| 256 | 1.2843x |

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
