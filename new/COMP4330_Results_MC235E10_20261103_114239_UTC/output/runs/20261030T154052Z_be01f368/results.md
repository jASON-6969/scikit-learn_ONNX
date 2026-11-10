# CPU inference benchmark results

This run compares the same trained 100-tree Random Forest on the Breast Cancer Wisconsin dataset using scikit-learn and two ONNX Runtime output representations. All measurements come from this run on one CPU machine. Hardware and software versions are recorded in [environment.json](environment.json); the data split is recorded in [dataset.json](dataset.json), model parameters in [model_parameters.json](model_parameters.json), effective runtime and thread settings in [runtime_settings.json](runtime_settings.json), and experiment and timing settings in [config.json](config.json).

## All measured configurations

| Backend | Batch size | Timed calls | Median (ms) | P25-P75 (ms) | P95 (ms) | Throughput (samples/s) | Speedup vs scikit-learn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scikit-learn | 1 | 1000 | 4.6051 | 3.59855-5.16483 | 7.2959 | 217.151 | 1.0000x |
| scikit-learn | 16 | 1000 | 4.38445 | 3.36273-4.92373 | 7.45182 | 3649.26 | 1.0000x |
| scikit-learn | 64 | 1000 | 4.76555 | 3.6477-5.38715 | 8.61608 | 13429.7 | 1.0000x |
| scikit-learn | 256 | 1000 | 5.16 | 4.06827-5.73835 | 6.93837 | 49612.4 | 1.0000x |
| ONNX Runtime (ZipMap on) | 1 | 1000 | 0.01305 | 0.0111-0.0158 | 0.02021 | 76628.4 | 352.8812x |
| ONNX Runtime (ZipMap on) | 16 | 1000 | 0.0438 | 0.0403-0.0456 | 0.05634 | 365297 | 100.1016x |
| ONNX Runtime (ZipMap on) | 64 | 1000 | 0.091 | 0.0873-0.142325 | 0.17464 | 703297 | 52.3687x |
| ONNX Runtime (ZipMap on) | 256 | 1000 | 0.6049 | 0.4679-0.674 | 0.90222 | 423210 | 8.5303x |
| ONNX Runtime (ZipMap off) | 1 | 1000 | 0.0128 | 0.0097-0.0137 | 0.0155 | 78125 | 359.7734x |
| ONNX Runtime (ZipMap off) | 16 | 1000 | 0.0352 | 0.0241-0.0374 | 0.049255 | 454545 | 124.5582x |
| ONNX Runtime (ZipMap off) | 64 | 1000 | 0.0778 | 0.0734-0.1205 | 0.23148 | 822622 | 61.2539x |
| ONNX Runtime (ZipMap off) | 256 | 1000 | 0.48845 | 0.397-0.571 | 0.71652 | 524107 | 10.5640x |

The median is the middle observed per-call timing. P25-P75 shows the central 50% of timings (the interquartile range); P95 is the 95th percentile. These describe timing variation and are not confidence intervals for a population of machines. Throughput is batch size divided by median call time. Speedup is the scikit-learn median divided by the compared backend median for the same batch size: above 1 means faster and below 1 means slower.

![Median latency and interquartile range](figures/latency.png)

![Throughput from median latency](figures/throughput.png)

## ZipMap comparison

| Batch size | Median latency: ZipMap on / ZipMap off |
| ---: | ---: |
| 1 | 1.0195x |
| 16 | 1.2443x |
| 64 | 1.1697x |
| 256 | 1.2384x |

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
