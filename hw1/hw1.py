import ctypes
import sys
import time
from functools import lru_cache
from pathlib import Path

import numba
import numpy as np


def make_timestamps(fps: int, st_ts: float, fn_ts: float) -> np.ndarray:
    """
    Create array of timestamps. This array is discretized with fps,
    but not evenly.
    Timestamps are assumed sorted nad unique.
    Parameters:
    - fps: int
        Average frame per second
    - st_ts: float
        First timestamp in the sequence
    - fn_ts: float
        Last timestamp in the sequence
    Returns:
        np.ndarray: synthetic timestamps
    """
    # generate uniform timestamps
    timestamps = np.linspace(st_ts, fn_ts, int((fn_ts - st_ts) * fps))
    # add an fps noise
    timestamps += np.random.randn(len(timestamps))
    timestamps = np.unique(np.sort(timestamps))
    return timestamps


def match_timestamps_numpy(
    timestamps1: np.ndarray, timestamps2: np.ndarray
) -> np.ndarray:
    """
    Timestamp matching function. It returns such array `matching` of length len(timestamps1),
    that for each index i of timestamps1 the output element matching[i] contains
    the index j of timestamps2, so that the difference between
    timestamps2[j] and timestamps1[i] is minimal.
    """
    matching = np.zeros(len(timestamps1), dtype=np.intp)
    if len(timestamps1) == 0:
        return matching
    if len(timestamps2) == 0:
        raise ValueError("timestamps2 must contain at least one timestamp")

    if timestamps1[-1] < timestamps2[0]:
        return matching
    if timestamps1[0] > timestamps2[-1]:
        matching.fill(len(timestamps2) - 1)
        return matching

    overlap_start = max(timestamps1[0], timestamps2[0])
    overlap_end = min(timestamps1[-1], timestamps2[-1])

    start = np.searchsorted(timestamps1, overlap_start, side="left")
    end = np.searchsorted(timestamps1, overlap_end, side="right")
    in_overlap = timestamps1[start:end]

    right = np.searchsorted(timestamps2, in_overlap, side="left")
    left = np.maximum(right - 1, 0)
    right = np.minimum(right, len(timestamps2) - 1)

    use_right = timestamps2[right] - in_overlap < in_overlap - timestamps2[left]
    matching[start:end] = np.where(use_right, right, left)
    matching[end:] = len(timestamps2) - 1
    return matching


@numba.njit
def match_timestamps(
    timestamps1: np.ndarray, timestamps2: np.ndarray
) -> np.ndarray:
    """
    Timestamp matching function. It returns such array `matching` of length len(timestamps1),
    that for each index i of timestamps1 the output element matching[i] contains
    the index j of timestamps2, so that the difference between
    timestamps2[j] and timestamps1[i] is minimal.
    """
    matching = np.zeros(len(timestamps1), dtype=np.intp)
    if len(timestamps1) == 0:
        return matching
    if len(timestamps2) == 0:
        raise ValueError("timestamps2 must contain at least one timestamp")

    if timestamps1[-1] < timestamps2[0]:
        return matching
    if timestamps1[0] > timestamps2[-1]:
        matching[:] = len(timestamps2) - 1
        return matching

    overlap_start = max(timestamps1[0], timestamps2[0])
    overlap_end = min(timestamps1[-1], timestamps2[-1])

    start = np.searchsorted(timestamps1, overlap_start, side="left")
    end = np.searchsorted(timestamps1, overlap_end, side="right")
    j = max(0, np.searchsorted(timestamps2, timestamps1[start], side="left") - 1)

    for i in range(start, end):
        timestamp = timestamps1[i]
        while j + 1 < len(timestamps2) and timestamps2[j + 1] <= timestamp:
            j += 1

        if (j + 1 < len(timestamps2) and timestamps2[j + 1] - timestamp < timestamp - timestamps2[j]):
            matching[i] = j + 1
        else:
            matching[i] = j

    for i in range(end, len(timestamps1)):
        matching[i] = len(timestamps2) - 1

    return matching


@lru_cache(maxsize=1)
def _load_cpp_library():
    if sys.platform == "win32":
        library_name = "match_timestamps.dll"
    elif sys.platform == "darwin":
        library_name = "libmatch_timestamps.dylib"
    else:
        library_name = "libmatch_timestamps.so"

    library_path = Path(__file__).with_name(library_name)
    if not library_path.is_file():
        raise FileNotFoundError(
            f"Native library not found: {library_path}. "
            "Build it with the command in README.md."
        )

    library = ctypes.CDLL(str(library_path))
    function = library.match_timestamps_cpp
    double_pointer = ctypes.POINTER(ctypes.c_double)
    index_pointer = ctypes.POINTER(ctypes.c_ssize_t)
    function.argtypes = [
        double_pointer,
        ctypes.c_ssize_t,
        double_pointer,
        ctypes.c_ssize_t,
        index_pointer,
    ]
    function.restype = None
    return function


def match_timestamps_cpp(
    timestamps1: np.ndarray, timestamps2: np.ndarray
) -> np.ndarray:
    if len(timestamps1) == 0:
        return np.empty(0, dtype=np.intp)
    if len(timestamps2) == 0:
        raise ValueError("timestamps2 must contain at least one timestamp")

    timestamps1 = np.ascontiguousarray(timestamps1, dtype=np.float64)
    timestamps2 = np.ascontiguousarray(timestamps2, dtype=np.float64)
    matching = np.zeros(len(timestamps1), dtype=np.intp)
    _load_cpp_library()(
        timestamps1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(timestamps1),
        timestamps2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(timestamps2),
        matching.ctypes.data_as(ctypes.POINTER(ctypes.c_ssize_t)),
    )
    return matching


def benchmark(function, timestamps1, timestamps2, repeats=5):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function(timestamps1, timestamps2)
        samples.append(time.perf_counter() - start)
    return sorted(samples)[len(samples) // 2]


def main():
    """
    Setup:
        Say we have two cameras, each filming the same scene. We make
        a prediction based on this scene (e.g. detect a human pose).
        To improve the robustness of the detection algorithm,
        we average the predictions from both cameras at each moment.
        The camera data is a pair (frame, timestamp), where the timestamp
        represents the moment when the frame was captured by the camera.
    Problem:
        For each frame of camera1, we need to find the index of the
        corresponding frame received by camera2. The frame i from camera2
        corresponds to the frame j from camera1, if
        abs(timestamps[i] - timestamps[j]) is minimal for all i.
    """
    # generate timestamps for the first camera
    timestamps1 = make_timestamps(30, time.time(), time.time() + 3600)
    # generate timestamps for the second camera
    timestamps2 = make_timestamps(60, time.time(), time.time() + 3600)
    numpy_matching = match_timestamps_numpy(timestamps1, timestamps2)

    start = time.perf_counter()
    numba_matching = match_timestamps(timestamps1, timestamps2)
    numba_first_call = time.perf_counter() - start

    cpp_matching = match_timestamps_cpp(timestamps1, timestamps2)
    if not np.array_equal(numpy_matching, numba_matching):
        raise RuntimeError("NumPy and Numba matching results differ")
    if not np.array_equal(numpy_matching, cpp_matching):
        raise RuntimeError("NumPy and C++ matching results differ")

    numpy_time = benchmark(match_timestamps_numpy, timestamps1, timestamps2)
    numba_time = benchmark(match_timestamps, timestamps1, timestamps2)
    cpp_time = benchmark(match_timestamps_cpp, timestamps1, timestamps2)

    print(f"NumPy current (median of 5): {numpy_time * 1000:.3f} ms")
    print(f"Numba warm (median of 5): {numba_time * 1000:.3f} ms")
    print(f"C++ native (median of 5): {cpp_time * 1000:.3f} ms")
    print(f"Numba first call (incl. compilation): {numba_first_call * 1000:.3f} ms")
    print(f"Fastest warm implementation: {min(numpy_time, numba_time, cpp_time) * 1000:.3f} ms")


if __name__ == '__main__':
    main()
