# CPU inference benchmark results

This run compares the same trained 500-tree Random Forest on the Breast Cancer Wisconsin dataset using scikit-learn and two ONNX Runtime output representations. All measurements come from this run on one CPU machine. Hardware and software versions are recorded in [environment.json](environment.json); the data split is recorded in [dataset.json](dataset.json), model parameters in [model_parameters.json](model_parameters.json), effective runtime and thread settings in [runtime_settings.json](runtime_settings.json), and experiment and timing settings in [config.json](config.json).

## All measured configurations

| Backend | Batch size | Timed calls | Median (ms) | P25-P75 (ms) | P95 (ms) | Throughput (samples/s) | Speedup vs scikit-learn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scikit-learn | 1 | 1000 | 20.9566 | 17.7943-22.7907 | 29.5784 | 47.7178 | 1.0000x |
| scikit-learn | 16 | 1000 | 21.386 | 18.1431-23.3904 | 30.1564 | 748.153 | 1.0000x |
| scikit-learn | 64 | 1000 | 22.43 | 19.3745-24.6732 | 33.7443 | 2853.33 | 1.0000x |
| scikit-learn | 256 | 1000 | 24.1499 | 20.919-26.4746 | 33.0062 | 10600.5 | 1.0000x |
| ONNX Runtime (ZipMap on) | 1 | 1000 | 0.0242 | 0.0169-0.0253 | 0.0297 | 41322.3 | 865.9731x |
| ONNX Runtime (ZipMap on) | 16 | 1000 | 0.14425 | 0.107275-0.1594 | 0.21276 | 110919 | 148.2565x |
| ONNX Runtime (ZipMap on) | 64 | 1000 | 0.9414 | 0.7943-1.0123 | 1.20075 | 67983.9 | 23.8262x |
| ONNX Runtime (ZipMap on) | 256 | 1000 | 3.6433 | 3.1869-3.89885 | 5.27197 | 70266 | 6.6286x |
| ONNX Runtime (ZipMap off) | 1 | 1000 | 0.0201 | 0.0153-0.0226 | 0.025105 | 49751.2 | 1042.6144x |
| ONNX Runtime (ZipMap off) | 16 | 1000 | 0.1359 | 0.0988-0.1518 | 0.22304 | 117734 | 157.3657x |
| ONNX Runtime (ZipMap off) | 64 | 1000 | 0.92115 | 0.770775-0.998875 | 1.6904 | 69478.4 | 24.3499x |
| ONNX Runtime (ZipMap off) | 256 | 1000 | 3.5124 | 2.98045-3.75195 | 4.10515 | 72884.6 | 6.8756x |

The median is the middle observed per-call timing. P25-P75 shows the central 50% of timings (the interquartile range); P95 is the 95th percentile. These describe timing variation and are not confidence intervals for a population of machines. Throughput is batch size divided by median call time. Speedup is the scikit-learn median divided by the compared backend median for the same batch size: above 1 means faster and below 1 means slower.

![Median latency and interquartile range](figures/latency.png)

![Throughput from median latency](figures/throughput.png)

## ZipMap comparison

| Batch size | Median latency: ZipMap on / ZipMap off |
| ---: | ---: |
| 1 | 1.2040x |
| 16 | 1.0614x |
| 64 | 1.0220x |
| 256 | 1.0373x |

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
