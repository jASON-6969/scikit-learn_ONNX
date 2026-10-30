# CPU inference benchmark results

This run compares the same trained 100-tree Random Forest on the Breast Cancer Wisconsin dataset using scikit-learn and two ONNX Runtime output representations. All measurements come from this run on one CPU machine. Hardware and software versions are recorded in [environment.json](environment.json); the data split is recorded in [dataset.json](dataset.json), model parameters in [model_parameters.json](model_parameters.json), effective runtime and thread settings in [runtime_settings.json](runtime_settings.json), and experiment and timing settings in [config.json](config.json).

## All measured configurations

| Backend | Batch size | Timed calls | Median (ms) | P25-P75 (ms) | P95 (ms) | Throughput (samples/s) | Speedup vs scikit-learn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scikit-learn | 1 | 1000 | 4.6844 | 3.17157-6.9207 | 12.1951 | 213.475 | 1.0000x |
| scikit-learn | 16 | 1000 | 2.61425 | 2.34353-3.22697 | 4.10991 | 6120.3 | 1.0000x |
| scikit-learn | 64 | 1000 | 3.232 | 2.49927-3.87648 | 5.21484 | 19802 | 1.0000x |
| scikit-learn | 256 | 1000 | 6.37035 | 5.18602-8.8037 | 11.5409 | 40186.2 | 1.0000x |
| ONNX Runtime (ZipMap on) | 1 | 1000 | 0.01075 | 0.0079-0.0135 | 0.04113 | 93023.3 | 435.7581x |
| ONNX Runtime (ZipMap on) | 16 | 1000 | 0.0196 | 0.0195-0.0205 | 0.0397 | 816327 | 133.3801x |
| ONNX Runtime (ZipMap on) | 64 | 1000 | 0.0896 | 0.0614-0.106225 | 0.177315 | 714286 | 36.0714x |
| ONNX Runtime (ZipMap on) | 256 | 1000 | 0.70925 | 0.511525-1.1034 | 1.59343 | 360945 | 8.9818x |
| ONNX Runtime (ZipMap off) | 1 | 1000 | 0.0097 | 0.007575-0.0134 | 0.03623 | 103093 | 482.9278x |
| ONNX Runtime (ZipMap off) | 16 | 1000 | 0.0166 | 0.0164-0.0172 | 0.027605 | 963855 | 157.4849x |
| ONNX Runtime (ZipMap off) | 64 | 1000 | 0.0762 | 0.0515-0.0865 | 0.10982 | 839895 | 42.4147x |
| ONNX Runtime (ZipMap off) | 256 | 1000 | 0.64045 | 0.4192-0.97515 | 1.39473 | 399719 | 9.9467x |

The median is the middle observed per-call timing. P25-P75 shows the central 50% of timings (the interquartile range); P95 is the 95th percentile. These describe timing variation and are not confidence intervals for a population of machines. Throughput is batch size divided by median call time. Speedup is the scikit-learn median divided by the compared backend median for the same batch size: above 1 means faster and below 1 means slower.

![Median latency and interquartile range](figures/latency.png)

![Throughput from median latency](figures/throughput.png)

## ZipMap comparison

| Batch size | Median latency: ZipMap on / ZipMap off |
| ---: | ---: |
| 1 | 1.1082x |
| 16 | 1.1807x |
| 64 | 1.1759x |
| 256 | 1.1074x |

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
