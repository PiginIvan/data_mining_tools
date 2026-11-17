#include <algorithm>
#include <cstdint>

extern "C" void match_timestamps_cpp(
    const double* timestamps1,
    std::intptr_t len1,
    const double* timestamps2,
    std::intptr_t len2,
    std::intptr_t* matching
) {
    if (len1 == 0) {
        return;
    }
    std::fill(matching, matching + len1, 0);
    if (len2 == 0) {
        return;
    }

    const double overlap_start = std::max(timestamps1[0], timestamps2[0]);
    const double overlap_end = std::min(timestamps1[len1 - 1], timestamps2[len2 - 1]);
    if (overlap_start > overlap_end) {
        return;
    }

    const double* start_it = std::lower_bound(timestamps1, timestamps1 + len1, overlap_start);
    const double* end_it = std::upper_bound(timestamps1, timestamps1 + len1, overlap_end);
    const std::intptr_t start = start_it - timestamps1;
    const std::intptr_t end = end_it - timestamps1;

    const double* right_it = std::lower_bound(timestamps2, timestamps2 + len2, timestamps1[start]);
    std::intptr_t j = right_it - timestamps2;
    if (j > 0) {
        --j;
    }

    for (std::intptr_t i = start; i < end; ++i) {
        const double timestamp = timestamps1[i];
        while (j + 1 < len2 && timestamps2[j + 1] <= timestamp) {
            ++j;
        }

        if (j + 1 < len2 && timestamps2[j + 1] - timestamp < timestamp - timestamps2[j]) {
            matching[i] = j + 1;
        } else {
            matching[i] = j;
        }
    }
}
