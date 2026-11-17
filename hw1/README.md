## HW1. Matching timestamps.
## Author: Pigin Ivan

### Usage:
```bash
docker build -t hw1-timestamp-match ./hw1
```

```bash
docker run --rm hw1-timestamp-match
```

### Benchmark

| Implementation | Time |

| NumPy | 8.938 ms |

| Numba, warm | 2.262 ms |

| C++ native | 1.531 ms |

| Numba, first call including compilation | 1848.576 ms |

Best result: C++ native, `1.531ms`
