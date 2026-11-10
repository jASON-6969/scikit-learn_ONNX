# CPU inference benchmark results

This run compares the same trained 10-tree Random Forest on the Breast Cancer Wisconsin dataset using scikit-learn and two ONNX Runtime output representations. All measurements come from this run on one CPU machine. Hardware and software versions are recorded in [environment.json](environment.json); the data split is recorded in [dataset.json](dataset.json), model parameters in [model_parameters.json](model_parameters.json), effective runtime and thread settings in [runtime_settings.json](runtime_settings.json), and experiment and timing settings in [config.json](config.json).

## All measured configurations

| Backend | Batch size | Timed calls | Median (ms) | P25-P75 (ms) | P95 (ms) | Throughput (samples/s) | Speedup vs scikit-learn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scikit-learn | 1 | 1000 | 0.581 | 0.44655-0.6656 | 1.04078 | 1721.17 | 1.0000x |
| scikit-learn | 16 | 1000 | 0.58905 | 0.453825-0.703625 | 1.17696 | 27162.4 | 1.0000x |
| scikit-learn | 64 | 1000 | 0.6062 | 0.460775-0.6626 | 0.90821 | 105576 | 1.0000x |
| scikit-learn | 256 | 1000 | 0.66125 | 0.482475-0.748225 | 1.01368 | 387146 | 1.0000x |
| ONNX Runtime (ZipMap on) | 1 | 1000 | 0.0098 | 0.0097-0.0129 | 0.0152 | 102041 | 59.2857x |
| ONNX Runtime (ZipMap on) | 16 | 1000 | 0.015 | 0.0145-0.021 | 0.0231 | 1.06667e+06 | 39.2700x |
| ONNX Runtime (ZipMap on) | 64 | 1000 | 0.0296 | 0.0292-0.041725 | 0.048405 | 2.16216e+06 | 20.4797x |
| ONNX Runtime (ZipMap on) | 256 | 1000 | 0.0942 | 0.089-0.133125 | 0.15932 | 2.71762e+06 | 7.0196x |
| ONNX Runtime (ZipMap off) | 1 | 1000 | 0.0083 | 0.0082-0.0111 | 0.012515 | 120482 | 70.0000x |
| ONNX Runtime (ZipMap off) | 16 | 1000 | 0.0108 | 0.0103-0.0146 | 0.0161 | 1.48148e+06 | 54.5417x |
| ONNX Runtime (ZipMap off) | 64 | 1000 | 0.0161 | 0.0158-0.023 | 0.02835 | 3.97516e+06 | 37.6522x |
| ONNX Runtime (ZipMap off) | 256 | 1000 | 0.0368 | 0.0364-0.0564 | 0.06481 | 6.95652e+06 | 17.9688x |

The median is the middle observed per-call timing. P25-P75 shows the central 50% of timings (the interquartile range); P95 is the 95th percentile. These describe timing variation and are not confidence intervals for a population of machines. Throughput is batch size divided by median call time. Speedup is the scikit-learn median divided by the compared backend median for the same batch size: above 1 means faster and below 1 means slower.

![Median latency and interquartile range](figures/latency.png)

![Throughput from median latency](figures/throughput.png)

## ZipMap comparison

| Batch size | Median latency: ZipMap on / ZipMap off |
| ---: | ---: |
| 1 | 1.1807x |
| 16 | 1.3889x |
| 64 | 1.8385x |
| 256 | 2.5598x |

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
