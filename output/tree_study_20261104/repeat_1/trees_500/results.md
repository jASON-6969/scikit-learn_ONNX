# CPU inference benchmark results

This run compares the same trained 500-tree Random Forest on the Breast Cancer Wisconsin dataset using scikit-learn and two ONNX Runtime output representations. All measurements come from this run on one CPU machine. Hardware and software versions are recorded in [environment.json](environment.json); the data split is recorded in [dataset.json](dataset.json), model parameters in [model_parameters.json](model_parameters.json), effective runtime and thread settings in [runtime_settings.json](runtime_settings.json), and experiment and timing settings in [config.json](config.json).

## All measured configurations

| Backend | Batch size | Timed calls | Median (ms) | P25-P75 (ms) | P95 (ms) | Throughput (samples/s) | Speedup vs scikit-learn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scikit-learn | 1 | 1000 | 21.0393 | 17.8211-22.9554 | 31.1962 | 47.53 | 1.0000x |
| scikit-learn | 16 | 1000 | 21.4327 | 18.2872-23.2952 | 30.1146 | 746.521 | 1.0000x |
| scikit-learn | 64 | 1000 | 22.5386 | 19.2302-24.335 | 31.9167 | 2839.58 | 1.0000x |
| scikit-learn | 256 | 1000 | 24.5323 | 21.2448-26.9187 | 34.6982 | 10435.2 | 1.0000x |
| ONNX Runtime (ZipMap on) | 1 | 1000 | 0.01755 | 0.0165-0.0246 | 0.02921 | 56980.1 | 1198.8234x |
| ONNX Runtime (ZipMap on) | 16 | 1000 | 0.15205 | 0.11115-0.1687 | 0.353 | 105229 | 140.9586x |
| ONNX Runtime (ZipMap on) | 64 | 1000 | 0.9405 | 0.800475-1.01225 | 1.25455 | 68048.9 | 23.9644x |
| ONNX Runtime (ZipMap on) | 256 | 1000 | 3.62945 | 3.22843-3.82393 | 4.42291 | 70534.1 | 6.7592x |
| ONNX Runtime (ZipMap off) | 1 | 1000 | 0.02 | 0.0151-0.0232 | 0.028005 | 50000 | 1051.9675x |
| ONNX Runtime (ZipMap off) | 16 | 1000 | 0.1442 | 0.120875-0.158225 | 0.183745 | 110957 | 148.6321x |
| ONNX Runtime (ZipMap off) | 64 | 1000 | 0.90645 | 0.77295-0.993925 | 1.47132 | 70605.1 | 24.8646x |
| ONNX Runtime (ZipMap off) | 256 | 1000 | 3.522 | 2.9869-3.69647 | 4.86334 | 72686 | 6.9654x |

The median is the middle observed per-call timing. P25-P75 shows the central 50% of timings (the interquartile range); P95 is the 95th percentile. These describe timing variation and are not confidence intervals for a population of machines. Throughput is batch size divided by median call time. Speedup is the scikit-learn median divided by the compared backend median for the same batch size: above 1 means faster and below 1 means slower.

![Median latency and interquartile range](figures/latency.png)

![Throughput from median latency](figures/throughput.png)

## ZipMap comparison

| Batch size | Median latency: ZipMap on / ZipMap off |
| ---: | ---: |
| 1 | 0.8775x |
| 16 | 1.0544x |
| 64 | 1.0376x |
| 256 | 1.0305x |

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
