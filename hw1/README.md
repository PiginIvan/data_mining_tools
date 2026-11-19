## HW1. Matching timestamps.
## Author: Pigin Ivan

### Usage:
```bash
git clone https://github.com/PiginIvan/data_mining_tools.git
```

```bash
cd hw1
```

```bash
docker build -t hw1-timestamp-match ./hw1
```

```bash
docker run --rm hw1-timestamp-match
```

### Benchmark

| Implementation | Time |
| --- | ---: |
| NumPy | 6.877 ms |
| Numba, warm | 1.770 ms |
| C++ native | 1.305 ms |
| Numba, first call including compilation | 1422.544 ms |

Best warm result: C++ native, `1.305 ms`.
