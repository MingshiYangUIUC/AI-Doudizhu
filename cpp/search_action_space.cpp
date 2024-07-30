#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>  // Include this header for automatic conversions
#include <vector>
#include <utility>
#include <string>
#include <iostream> // For std::cout
#include <algorithm> // for std::next_permutation
#include <set>       // for std::set

namespace py = pybind11;

std::tuple<std::vector<std::pair<std::string, std::pair<int, int>>>, std::vector<std::pair<int, int>>> get_all_actions(py::array_t<int> input_array) {
    // Convert numpy array to C++ vector
    py::buffer_info buf = input_array.request();
    int *ptr = static_cast<int *>(buf.ptr);
    int size = static_cast<int>(buf.size);

    // Vector to store the counts
    std::vector<int> counts(ptr, ptr + size);

    // Vector to store the pairs
    std::vector<std::pair<std::string, std::pair<int, int>>> outputs;

    // String to store the pair label
    std::string action_str;

    // Custom labels for the cards
    const std::string labels = "3456789XJQKA2BR";

    // Pass
    // outputs.emplace_back("", std::make_pair(0, 0));

    // Find all singles
    for (int i = 0; i < 15; ++i) {
        if (counts[i] > 0) {
            action_str = labels[i];
            outputs.emplace_back(action_str, std::make_pair(1, i));
        }
    }

    // Find all doubles
    for (int i = 0; i < 15; ++i) {
        if (counts[i] > 1) {
            action_str = {labels[i], labels[i]};
            outputs.emplace_back(action_str, std::make_pair(2, i));
        }
    }

    // Find all triples
    std::vector<int> idx_tri;
    for (int i = 0; i < 15; ++i) {
        if (counts[i] > 2) {
            idx_tri.push_back(i);
            action_str = {labels[i], labels[i], labels[i]};
            outputs.emplace_back(action_str, std::make_pair(3, i));
        }
    }

    // Find all bombs
    std::vector<int> idx_quad;
    for (int i = 0; i < (15-2); ++i) {
        if (counts[i] == 4) {
            idx_quad.push_back(i);
            action_str = {labels[i], labels[i], labels[i], labels[i]};
            outputs.emplace_back(action_str, std::make_pair(4, i));
        }
    }
        // add king bomb
    if (counts[13] == 1 && counts[14] == 1) {
        action_str = {labels[13], labels[14]};
        outputs.emplace_back(action_str, std::make_pair(4, 13));
    }

    // Find 3 + 1 and 3 + 2
    for (int idx : idx_tri) {
        for (int j = 0; j < 15; ++j) {
            // Find all singles
            if (idx != j && counts[j] > 0) {
                action_str = {labels[idx], labels[idx], labels[idx], labels[j]};
                outputs.emplace_back(action_str, std::make_pair(5, idx));
            }
            // Find all doubles
            if (idx != j && counts[j] > 1) {
                action_str = {labels[idx], labels[idx], labels[idx], labels[j], labels[j]};
                outputs.emplace_back(action_str, std::make_pair(6, idx));
            }
        }
    }

    // Find 4 + 1 + 1 and 4 + 2 + 2
    for (int idx : idx_quad) {
        // Find all singles
        for (int j = 0; j < 15; ++j) {
            for (int k = 0; k < j + 1; ++k) {
                // Find all singles again
                if (idx != j && idx != k && counts[j] > 0 && counts[k] > 0 && (j != k || counts[j] > 1)) {
                    action_str = {labels[idx], labels[idx], labels[idx], labels[idx], labels[j], labels[k]};
                    outputs.emplace_back(action_str, std::make_pair(7, idx));
                }
                // Find all doubles again
                if (idx != j && idx != k && counts[j] > 1 && counts[k] > 1 && (j != k || counts[j] == 4)) {
                    action_str = {labels[idx], labels[idx], labels[idx], labels[idx], labels[j], labels[j], labels[k], labels[k]};
                    outputs.emplace_back(action_str, std::make_pair(8, idx));
                }
            }
        }
    }

    // Find x1 5+ sequence
    for (int i = 0; i < 8; ++i) {
        int l = 5;
        bool invalid = false;

        // Check if there is a zero
        for (int j = i; j < i + l; ++j) {
            if (counts[j] == 0) {
                invalid = true;
                break;
            }
        }

        // Construct sequence
        if (!invalid) {
            action_str = "";
            for (int k = i; k < i + l; ++k) {
                action_str += labels[k];
            }
            outputs.emplace_back(action_str, std::make_pair(9, i));
            while ((i + l < 12) && (counts[i + l] != 0)) {
                l += 1;
                action_str = "";
                for (int k = i; k < i + l; ++k) {
                    action_str += labels[k];
                }
                outputs.emplace_back(action_str, std::make_pair(9, i));
            }
        }
    }

    // Find x2 3+ sequence
    for (int i = 0; i < 10; ++i) {
        int l = 3;
        bool invalid = false;

        // Check
        for (int j = i; j < i + l; ++j) {
            if (counts[j] < 2) {
                invalid = true;
                break;
            }
        }

        // Construct sequence
        if (!invalid) {
            action_str = "";
            for (int k = i; k < i + l; ++k) {
                action_str += labels[k];
                action_str += labels[k];
            }
            outputs.emplace_back(action_str, std::make_pair(10, i));
            while ((i + l < 12) && (counts[i + l] >= 2)) {
                l += 1;
                action_str = "";
                for (int k = i; k < i + l; ++k) {
                    action_str += labels[k];
                    action_str += labels[k];
                }
                outputs.emplace_back(action_str, std::make_pair(10, i));
            }
        }
    }

    // Find x3 2+ sequence
    std::vector<std::pair<int, int>> i_l_triseq;
    for (int i = 0; i < 11; ++i) {
        int l = 2;
        bool invalid = false;

        // Check
        for (int j = i; j < i + l; ++j) {
            if (counts[j] < 3) {
                invalid = true;
                break;
            }
        }

        // Construct sequence
        if (!invalid) {
            action_str = "";
            for (int k = i; k < i + l; ++k) {
                action_str += labels[k];
                action_str += labels[k];
                action_str += labels[k];
            }
            outputs.emplace_back(action_str, std::make_pair(11, i));
            i_l_triseq.push_back(std::make_pair(i, l));
            while ((i + l < 12) && (counts[i + l] >= 3)) {
                l += 1;
                action_str = "";
                for (int k = i; k < i + l; ++k) {
                    action_str += labels[k];
                    action_str += labels[k];
                    action_str += labels[k];
                }
                outputs.emplace_back(action_str, std::make_pair(11, i));
                i_l_triseq.push_back(std::make_pair(i, l));
            }
        }
    }
    return std::make_tuple(outputs, i_l_triseq);
}


std::vector<std::pair<std::string, std::pair<int, int>>> get_all_bombs(const std::vector<int>& counts, const std::string& labels) {
    std::vector<std::pair<std::string, std::pair<int, int>>> out;

    // Get quads
    for (int i = 0; i < 15 - 2; ++i) {
        if (counts[i] == 4) {
            std::string quad_label(4, labels[i]); // Construct string with 4 repeated characters
            out.emplace_back(quad_label, std::make_pair(4, i));
        }
    }

    // Check if there is BR
    if (counts[13] == 1 && counts[14] == 1) {
        out.emplace_back("BR", std::make_pair(4, 13));
    }

    return out;
}


std::vector<std::pair<std::string, std::pair<int, int>>> get_avail_actions(
    const std::string& input_string,
    const std::pair<int, int>& input_tuple,
    py::array_t<int> input_array
) {
    // Access the numpy array
    auto buf = input_array.request();
    int *ptr = static_cast<int *>(buf.ptr);
    int size = static_cast<int>(buf.size);
    
    // Vector to store the counts
    std::vector<int> counts(ptr, ptr + size);

    // Vector to store the pairs
    std::vector<std::pair<std::string, std::pair<int, int>>> outputs;

    // String to store the pair label
    std::string action_str;

    // Custom labels for the cards
    const std::string labels = "3456789XJQKA2BR";

    int input_info = input_tuple.first;
    int input_rank = input_tuple.second;
    int input_leng = static_cast<int>(input_string.length());

    // Singles
    if (input_info == 1) {
        for (int i = input_rank + 1; i < 15; ++i) {
            if (counts[i] > 0) {
                action_str = labels[i];
                outputs.emplace_back(action_str, std::make_pair(1, i));
            }
        }
        // Add bombs
        std::vector<std::pair<std::string, std::pair<int, int>>> bombs_output = get_all_bombs(counts, labels);
        outputs.insert(outputs.end(), bombs_output.begin(), bombs_output.end());
    }

    // Doubles
    else if (input_info == 2) {
        for (int i = input_rank + 1; i < 15 - 2; ++i) {
            if (counts[i] > 1) {
                action_str = {labels[i], labels[i]};
                outputs.emplace_back(action_str, std::make_pair(2, i));
            }
        }
        // Add bombs
        std::vector<std::pair<std::string, std::pair<int, int>>> bombs_output = get_all_bombs(counts, labels);
        outputs.insert(outputs.end(), bombs_output.begin(), bombs_output.end());
    }

    // Triples
    else if (input_info == 3) {
        for (int i = input_rank + 1; i < 15 - 2; ++i) {
            if (counts[i] > 2) {
                action_str = {labels[i], labels[i], labels[i]};
                outputs.emplace_back(action_str, std::make_pair(3, i));
            }
        }
        // Add bombs
        std::vector<std::pair<std::string, std::pair<int, int>>> bombs_output = get_all_bombs(counts, labels);
        outputs.insert(outputs.end(), bombs_output.begin(), bombs_output.end());
    }

    // Quads (make larger bombs and BR)
    else if (input_info == 4) {
        for (int i = input_rank + 1; i < 15 - 2; ++i) {
            if (counts[i] > 3) {
                action_str = {labels[i], labels[i], labels[i], labels[i]};
                outputs.emplace_back(action_str, std::make_pair(4, i));
            }
        }
        // Add King bomb
        if (counts[13] == 1 && counts[14] == 1) {
            outputs.emplace_back("BR", std::make_pair(4, 13));
        }
    }

    // 3 + 1
    else if (input_info == 5) {
        for (int i = input_rank + 1; i < 15 - 2; ++i) {
            if (counts[i] > 2) {
                for (int j = 0; j < 15; ++j) {
                    if (j != i && counts[j] > 0) {
                        action_str = {labels[i], labels[i], labels[i], labels[j]};
                        outputs.emplace_back(action_str, std::make_pair(5, i));
                    }
                }
            }
        }
        // Add bombs
        std::vector<std::pair<std::string, std::pair<int, int>>> bombs_output = get_all_bombs(counts, labels);
        outputs.insert(outputs.end(), bombs_output.begin(), bombs_output.end());
    }

    // 3 + 2
    else if (input_info == 6) {
        for (int i = input_rank + 1; i < 15 - 2; ++i) {
            if (counts[i] > 2) {
                for (int j = 0; j < 15 - 2; ++j) {
                    if (j != i && counts[j] > 1) {
                        action_str = {labels[i], labels[i], labels[i], labels[j], labels[j]};
                        outputs.emplace_back(action_str, std::make_pair(6, i));
                    }
                }
            }
        }
        // Add bombs
        std::vector<std::pair<std::string, std::pair<int, int>>> bombs_output = get_all_bombs(counts, labels);
        outputs.insert(outputs.end(), bombs_output.begin(), bombs_output.end());
    }

    // 4 + 1 + 1
    else if (input_info == 7) {
        for (int i = input_rank + 1; i < 15 - 2; ++i) {
            if (counts[i] > 3) {
                for (int j = 0; j < 15; ++j) {
                    for (int k = 0; k < j + 1; ++k) {
                        if (i != j && i!= k && counts[j] >= 1 && counts[k] >= 1 && (j != k || counts[j] >= 2)) {
                            action_str = {labels[i], labels[i], labels[i], labels[i], labels[j], labels[k]};
                            outputs.emplace_back(action_str, std::make_pair(7, i));
                        }
                    }
                }
            }
        }
        // Add bombs
        std::vector<std::pair<std::string, std::pair<int, int>>> bombs_output = get_all_bombs(counts, labels);
        outputs.insert(outputs.end(), bombs_output.begin(), bombs_output.end());
    }

    // 4 + 2 + 2
    else if (input_info == 8) {
        for (int i = input_rank + 1; i < 15 - 2; ++i) {
            if (counts[i] > 3) {
                for (int j = 0; j < 15; ++j) {
                    for (int k = 0; k < j + 1; ++k) {
                        if (i != j && i!= k && counts[j] >= 2 && counts[k] >= 2 && (j != k || counts[j] == 4)) {
                            action_str = {labels[i], labels[i], labels[i], labels[i], labels[j], labels[j], labels[k], labels[k]};
                            outputs.emplace_back(action_str, std::make_pair(8, i));
                        }
                    }
                }
            }
        }
        // Add bombs
        std::vector<std::pair<std::string, std::pair<int, int>>> bombs_output = get_all_bombs(counts, labels);
        outputs.insert(outputs.end(), bombs_output.begin(), bombs_output.end());
    }

    // 1 sequence
    else if (input_info == 9) {
        int l = input_leng;
        for (int i = input_rank + 1; i < 15 - 2 - l; ++i) {
            bool invalid = false;
            for (int j = i; j < i + l; ++j) {
                if (counts[j] == 0) {
                    invalid = true;
                }
            }
            if (!invalid) {
                action_str = "";
                for (int k = i; k < i + l; ++k) {
                    action_str += labels[k];
                }
                outputs.emplace_back(action_str, std::make_pair(9, i));
            }
        }
        // Add bombs
        std::vector<std::pair<std::string, std::pair<int, int>>> bombs_output = get_all_bombs(counts, labels);
        outputs.insert(outputs.end(), bombs_output.begin(), bombs_output.end());
    }

    // 2 sequence
    else if (input_info == 10) {
        int l = input_leng / 2;
        for (int i = input_rank + 1; i < 15 - 2 - l; ++i) {
            bool invalid = false;
            for (int j = i; j < i + l; ++j) {
                if (counts[j] < 2) {
                    invalid = true;
                }
            }
            if (!invalid) {
                action_str = "";
                for (int k = i; k < i + l; ++k) {
                    action_str += labels[k];
                    action_str += labels[k];
                }
                outputs.emplace_back(action_str, std::make_pair(10, i));
            }
        }
        // Add bombs
        std::vector<std::pair<std::string, std::pair<int, int>>> bombs_output = get_all_bombs(counts, labels);
        outputs.insert(outputs.end(), bombs_output.begin(), bombs_output.end());
    }

    // 3 sequence
    else if (input_info == 11) {
        int l = input_leng / 3;
        for (int i = input_rank + 1; i < 15 - 2 - l; ++i) {
            bool invalid = false;
            for (int j = i; j < i + l; ++j) {
                if (counts[j] < 3) {
                    invalid = true;
                }
            }
            if (!invalid) {
                action_str = "";
                for (int k = i; k < i + l; ++k) {
                    action_str += labels[k];
                    action_str += labels[k];
                    action_str += labels[k];
                }
                outputs.emplace_back(action_str, std::make_pair(11, i));
            }
        }
        // Add bombs
        std::vector<std::pair<std::string, std::pair<int, int>>> bombs_output = get_all_bombs(counts, labels);
        outputs.insert(outputs.end(), bombs_output.begin(), bombs_output.end());
    }

    return outputs;
}


std::tuple<std::vector<std::pair<py::array_t<int>, std::pair<int, int>>>, std::vector<std::pair<int, int>>> get_all_actions_array(py::array_t<int> input_array) {
    // Convert numpy array to C++ vector
    py::buffer_info buf = input_array.request();
    int *ptr = static_cast<int *>(buf.ptr);
    int size = static_cast<int>(buf.size);

    // Vector to store the counts
    std::vector<int> counts(ptr, ptr + size);

    // Vector to store the pairs
    std::vector<std::pair<py::array_t<int>, std::pair<int, int>>> outputs;

    // Vector to store the triple sequences
    std::vector<std::pair<int, int>> i_l_triseq;

    // Create an array part for reuse with all zeros
    py::array_t<int> action_arr = py::array_t<int>({size});
    auto buf_array = action_arr.request();
    int *array_ptr = static_cast<int *>(buf_array.ptr);
    std::fill(array_ptr, array_ptr + size, 0);  // Initialize all elements to zero initially

    // String to store the pair label
    // std::string action_str;

    // Custom labels for the cards
    // const std::string labels = "3456789XJQKA2BR";

    // Pass
    // outputs.emplace_back("", std::make_pair(0, 0));

    // Find all singles
    for (int i = 0; i < 15; ++i) {
        if (counts[i] > 0) {
            array_ptr[i] = 1;
            py::array_t<int> copy_arr = py::array_t<int>({size}, array_ptr);
            outputs.emplace_back(copy_arr, std::make_pair(1, i));
            array_ptr[i] = 0; // selective reset
        }
    }

    // Find all doubles
    for (int i = 0; i < 15; ++i) {
        if (counts[i] > 1) {
            array_ptr[i] = 2;
            py::array_t<int> copy_arr = py::array_t<int>({size}, array_ptr);
            outputs.emplace_back(copy_arr, std::make_pair(2, i));
            array_ptr[i] = 0; // selective reset
        }
    }

    // Find all triples
    std::vector<int> idx_tri;
    for (int i = 0; i < 15; ++i) {
        if (counts[i] > 2) {
            idx_tri.push_back(i);
            array_ptr[i] = 3;
            py::array_t<int> copy_arr = py::array_t<int>({size}, array_ptr);
            outputs.emplace_back(copy_arr, std::make_pair(3, i));
            array_ptr[i] = 0; // selective reset
        }
    }

    // Find all bombs
    std::vector<int> idx_quad;
    for (int i = 0; i < (15-2); ++i) {
        if (counts[i] == 4) {
            idx_quad.push_back(i);
            array_ptr[i] = 4;
            py::array_t<int> copy_arr = py::array_t<int>({size}, array_ptr);
            outputs.emplace_back(copy_arr, std::make_pair(4, i));
            array_ptr[i] = 0; // selective reset
        }
    }
        // add king bomb
    if (counts[13] == 1 && counts[14] == 1) {
        array_ptr[13] = 1;
        array_ptr[14] = 1;
        py::array_t<int> copy_arr = py::array_t<int>({size}, array_ptr);
        outputs.emplace_back(copy_arr, std::make_pair(4, 13));
        array_ptr[13] = 0; // selective reset
        array_ptr[14] = 0; // selective reset
    }

    // Find 3 + 1 and 3 + 2
    for (int idx : idx_tri) {
        for (int j = 0; j < 15; ++j) {
            // Find all singles
            if (idx != j && counts[j] > 0) {
                array_ptr[idx] = 3;
                array_ptr[j] = 1;
                py::array_t<int> copy_arr = py::array_t<int>({size}, array_ptr);
                outputs.emplace_back(copy_arr, std::make_pair(5, idx));
                array_ptr[idx] = 0;
                array_ptr[j] = 0;
            }
            // Find all doubles
            if (idx != j && counts[j] > 1) {
                array_ptr[idx] = 3;
                array_ptr[j] = 2;
                py::array_t<int> copy_arr = py::array_t<int>({size}, array_ptr);
                outputs.emplace_back(copy_arr, std::make_pair(6, idx));
                array_ptr[idx] = 0;
                array_ptr[j] = 0;
            }
        }
    }

    // Find 4 + 1 + 1 and 4 + 2 + 2
    for (int idx : idx_quad) {
        // Find all singles
        for (int j = 0; j < 15; ++j) {
            for (int k = 0; k < j + 1; ++k) {
                // Find all singles again
                if (idx != j && idx != k && counts[j] > 0 && counts[k] > 0 && (j != k || counts[j] > 1)) {
                    array_ptr[idx] = 4;
                    array_ptr[j] += 1;
                    array_ptr[k] += 1;
                    py::array_t<int> copy_arr = py::array_t<int>({size}, array_ptr);
                    outputs.emplace_back(copy_arr, std::make_pair(7, idx));
                    array_ptr[idx] = 0;
                    array_ptr[j] = 0;
                    array_ptr[k] = 0;
                }
                // Find all doubles again
                if (idx != j && idx != k && counts[j] > 1 && counts[k] > 1 && (j != k || counts[j] == 4)) {
                    array_ptr[idx] = 4;
                    array_ptr[j] += 2;
                    array_ptr[k] += 2;
                    py::array_t<int> copy_arr = py::array_t<int>({size}, array_ptr);
                    outputs.emplace_back(copy_arr, std::make_pair(8, idx));
                    array_ptr[idx] = 0;
                    array_ptr[j] = 0;
                    array_ptr[k] = 0;
                }
            }
        }
    }

    // Find x1 5+ sequence
    for (int i = 0; i < 8; ++i) {
        int l = 5;
        bool invalid = false;

        // Check if there is a zero
        for (int j = i; j < i + l; ++j) {
            if (counts[j] == 0) {
                invalid = true;
                break;
            }
        }

        // Construct sequence
        if (!invalid) {

            std::fill(array_ptr + i, array_ptr + i + l, 1);
            py::array_t<int> copy_arr = py::array_t<int>({size}, array_ptr);
            outputs.emplace_back(copy_arr, std::make_pair(9, i));
            std::fill(array_ptr + i, array_ptr + i + l, 0);

            while ((i + l < 12) && (counts[i + l] != 0)) {
                l += 1;

                std::fill(array_ptr + i, array_ptr + i + l, 1);
                py::array_t<int> copy_arr = py::array_t<int>({size}, array_ptr);
                outputs.emplace_back(copy_arr, std::make_pair(9, i));
                std::fill(array_ptr + i, array_ptr + i + l, 0);
            }
        }
    }

    // Find x2 3+ sequence
    for (int i = 0; i < 10; ++i) {
        int l = 3;
        bool invalid = false;

        // Check
        for (int j = i; j < i + l; ++j) {
            if (counts[j] < 2) {
                invalid = true;
                break;
            }
        }

        // Construct sequence
        if (!invalid) {

            std::fill(array_ptr + i, array_ptr + i + l, 2);
            py::array_t<int> copy_arr = py::array_t<int>({size}, array_ptr);
            outputs.emplace_back(copy_arr, std::make_pair(10, i));
            std::fill(array_ptr + i, array_ptr + i + l, 0);

            while ((i + l < 12) && (counts[i + l] >= 2)) {
                l += 1;

                std::fill(array_ptr + i, array_ptr + i + l, 2);
                py::array_t<int> copy_arr = py::array_t<int>({size}, array_ptr);
                outputs.emplace_back(copy_arr, std::make_pair(10, i));
                std::fill(array_ptr + i, array_ptr + i + l, 0);
            }
        }
    }

    // Find x3 2+ sequence
    for (int i = 0; i < 11; ++i) {
        int l = 2;
        bool invalid = false;

        // Check
        for (int j = i; j < i + l; ++j) {
            if (counts[j] < 3) {
                invalid = true;
                break;
            }
        }

        // Construct sequence
        if (!invalid) {

            std::fill(array_ptr + i, array_ptr + i + l, 3);
            py::array_t<int> copy_arr = py::array_t<int>({size}, array_ptr);
            outputs.emplace_back(copy_arr, std::make_pair(11, i));
            std::fill(array_ptr + i, array_ptr + i + l, 0);

            i_l_triseq.push_back(std::make_pair(i, l));
            while ((i + l < 12) && (counts[i + l] >= 3)) {
                l += 1;

                std::fill(array_ptr + i, array_ptr + i + l, 3);
                py::array_t<int> copy_arr = py::array_t<int>({size}, array_ptr);
                outputs.emplace_back(copy_arr, std::make_pair(11, i));
                std::fill(array_ptr + i, array_ptr + i + l, 0);

                i_l_triseq.push_back(std::make_pair(i, l));
            }
        }
    }
    return std::make_tuple(outputs, i_l_triseq);
}


std::vector<std::pair<py::array_t<int>, std::pair<int, int>>> get_all_bombs_array(const std::vector<int>& counts) {
    std::vector<std::pair<py::array_t<int>, std::pair<int, int>>> out;

    py::array_t<int> action_arr = py::array_t<int>({15});
    auto buf_array = action_arr.request();
    int *array_ptr = static_cast<int *>(buf_array.ptr);
    std::fill(array_ptr, array_ptr + 15, 0);  // Initialize all elements to zero initially

    // Get quads
    for (int i = 0; i < 15 - 2; ++i) {
        if (counts[i] == 4) {
            array_ptr[i] = 4;
            py::array_t<int> copy_arr = py::array_t<int>({15}, array_ptr);
            out.emplace_back(copy_arr, std::make_pair(4, i));
            array_ptr[i] = 0; // selective reset
        }
    }

    // Check if there is BR
    if (counts[13] == 1 && counts[14] == 1) {
        array_ptr[13] = 1;
        array_ptr[14] = 1;
        py::array_t<int> copy_arr = py::array_t<int>({15}, array_ptr);
        out.emplace_back(copy_arr, std::make_pair(4, 13));
        array_ptr[13] = 0; // selective reset
        array_ptr[14] = 0; // selective reset
    }

    return out;
}


std::vector<std::pair<py::array_t<int>, std::pair<int, int>>> get_avail_actions_array(
    int input_leng,
    const std::pair<int, int>& input_tuple,
    py::array_t<int> input_array
) {
    // Access the numpy array
    auto buf = input_array.request();
    int *ptr = static_cast<int *>(buf.ptr);
    int size = static_cast<int>(buf.size);
    
    // Vector to store the counts
    std::vector<int> counts(ptr, ptr + size);

    // Vector to store the pairs
    std::vector<std::pair<py::array_t<int>, std::pair<int, int>>> outputs;
    
    // Create an array part for reuse with all zeros
    py::array_t<int> action_arr = py::array_t<int>({size});
    auto buf_array = action_arr.request();
    int *array_ptr = static_cast<int *>(buf_array.ptr);
    std::fill(array_ptr, array_ptr + size, 0);  // Initialize all elements to zero initially

    // String to store the pair label
    // std::string action_str;

    // Custom labels for the cards
    // const std::string labels = "3456789XJQKA2BR";

    int input_info = input_tuple.first;
    int input_rank = input_tuple.second;
    // int input_leng = static_cast<int>(last_array.length());
    // input leng input directly

    // Singles
    if (input_info == 1) {
        for (int i = input_rank + 1; i < 15; ++i) {
            if (counts[i] > 0) {
                array_ptr[i] = 1;
                py::array_t<int> copy_arr = py::array_t<int>({size}, array_ptr);
                outputs.emplace_back(copy_arr, std::make_pair(1, i));
                array_ptr[i] = 0; // selective reset
            }
        }
        // Add bombs
        std::vector<std::pair<py::array_t<int>, std::pair<int, int>>> bombs_output = get_all_bombs_array(counts);
        outputs.insert(outputs.end(), bombs_output.begin(), bombs_output.end());
    }

    // Doubles
    else if (input_info == 2) {
        for (int i = input_rank + 1; i < 15 - 2; ++i) {
            if (counts[i] > 1) {
                array_ptr[i] = 2;
                py::array_t<int> copy_arr = py::array_t<int>({size}, array_ptr);
                outputs.emplace_back(copy_arr, std::make_pair(2, i));
                array_ptr[i] = 0; // selective reset
            }
        }
        // Add bombs
        std::vector<std::pair<py::array_t<int>, std::pair<int, int>>> bombs_output = get_all_bombs_array(counts);
        outputs.insert(outputs.end(), bombs_output.begin(), bombs_output.end());
    }

    // Triples
    else if (input_info == 3) {
        for (int i = input_rank + 1; i < 15 - 2; ++i) {
            if (counts[i] > 2) {
                array_ptr[i] = 3;
                py::array_t<int> copy_arr = py::array_t<int>({size}, array_ptr);
                outputs.emplace_back(copy_arr, std::make_pair(3, i));
                array_ptr[i] = 0; // selective reset
            }
        }
        // Add bombs
        std::vector<std::pair<py::array_t<int>, std::pair<int, int>>> bombs_output = get_all_bombs_array(counts);
        outputs.insert(outputs.end(), bombs_output.begin(), bombs_output.end());
    }

    // Quads (make larger bombs and BR)
    else if (input_info == 4) {
        for (int i = input_rank + 1; i < 15 - 2; ++i) {
            if (counts[i] > 3) {
                array_ptr[i] = 4;
                py::array_t<int> copy_arr = py::array_t<int>({size}, array_ptr);
                outputs.emplace_back(copy_arr, std::make_pair(4, i));
                array_ptr[i] = 0; // selective reset
            }
        }
        // Add King bomb
        if (counts[13] == 1 && counts[14] == 1) {
            array_ptr[13] = 1;
            array_ptr[14] = 1;
            py::array_t<int> copy_arr = py::array_t<int>({size}, array_ptr);
            outputs.emplace_back(copy_arr, std::make_pair(4, 13));
            array_ptr[13] = 0; // selective reset
            array_ptr[14] = 0; // selective reset
        }
    }

    // 3 + 1
    else if (input_info == 5) {
        for (int i = input_rank + 1; i < 15 - 2; ++i) {
            if (counts[i] > 2) {
                for (int j = 0; j < 15; ++j) {
                    if (j != i && counts[j] > 0) {
                        array_ptr[i] = 3;
                        array_ptr[j] += 1;
                        py::array_t<int> copy_arr = py::array_t<int>({size}, array_ptr);
                        outputs.emplace_back(copy_arr, std::make_pair(5, i));
                        array_ptr[i] = 0;
                        array_ptr[j] = 0;
                    }
                }
            }
        }
        // Add bombs
        std::vector<std::pair<py::array_t<int>, std::pair<int, int>>> bombs_output = get_all_bombs_array(counts);
        outputs.insert(outputs.end(), bombs_output.begin(), bombs_output.end());
    }

    // 3 + 2
    else if (input_info == 6) {
        for (int i = input_rank + 1; i < 15 - 2; ++i) {
            if (counts[i] > 2) {
                for (int j = 0; j < 15 - 2; ++j) {
                    if (j != i && counts[j] > 1) {
                        array_ptr[i] = 3;
                        array_ptr[j] += 2;
                        py::array_t<int> copy_arr = py::array_t<int>({size}, array_ptr);
                        outputs.emplace_back(copy_arr, std::make_pair(6, i));
                        array_ptr[i] = 0;
                        array_ptr[j] = 0;
                    }
                }
            }
        }
        // Add bombs
        std::vector<std::pair<py::array_t<int>, std::pair<int, int>>> bombs_output = get_all_bombs_array(counts);
        outputs.insert(outputs.end(), bombs_output.begin(), bombs_output.end());
    }

    // 4 + 1 + 1
    else if (input_info == 7) {
        for (int i = input_rank + 1; i < 15 - 2; ++i) {
            if (counts[i] > 3) {
                for (int j = 0; j < 15; ++j) {
                    for (int k = 0; k < j + 1; ++k) {
                        if (i != j && i!= k && counts[j] >= 1 && counts[k] >= 1 && (j != k || counts[j] >= 2)) {
                            array_ptr[i] = 4;
                            array_ptr[j] += 1;
                            array_ptr[k] += 1;
                            py::array_t<int> copy_arr = py::array_t<int>({size}, array_ptr);
                            outputs.emplace_back(copy_arr, std::make_pair(7, i));
                            array_ptr[i] = 0;
                            array_ptr[j] = 0;
                            array_ptr[k] = 0;
                        }
                    }
                }
            }
        }
        // Add bombs
        std::vector<std::pair<py::array_t<int>, std::pair<int, int>>> bombs_output = get_all_bombs_array(counts);
        outputs.insert(outputs.end(), bombs_output.begin(), bombs_output.end());
    }

    // 4 + 2 + 2
    else if (input_info == 8) {
        for (int i = input_rank + 1; i < 15 - 2; ++i) {
            if (counts[i] > 3) {
                for (int j = 0; j < 15; ++j) {
                    for (int k = 0; k < j + 1; ++k) {
                        if (i != j && i!= k && counts[j] >= 2 && counts[k] >= 2 && (j != k || counts[j] == 4)) {
                            array_ptr[i] = 4;
                            array_ptr[j] += 2;
                            array_ptr[k] += 2;
                            py::array_t<int> copy_arr = py::array_t<int>({size}, array_ptr);
                            outputs.emplace_back(copy_arr, std::make_pair(8, i));
                            array_ptr[i] = 0;
                            array_ptr[j] = 0;
                            array_ptr[k] = 0;
                        }
                    }
                }
            }
        }
        // Add bombs
        std::vector<std::pair<py::array_t<int>, std::pair<int, int>>> bombs_output = get_all_bombs_array(counts);
        outputs.insert(outputs.end(), bombs_output.begin(), bombs_output.end());
    }

    // 1 sequence
    else if (input_info == 9) {
        int l = input_leng;
        for (int i = input_rank + 1; i < 15 - 2 - l; ++i) {
            bool invalid = false;
            for (int j = i; j < i + l; ++j) {
                if (counts[j] == 0) {
                    invalid = true;
                }
            }
            if (!invalid) {
                std::fill(array_ptr + i, array_ptr + i + l, 1);
                py::array_t<int> copy_arr = py::array_t<int>({size}, array_ptr);
                outputs.emplace_back(copy_arr, std::make_pair(9, i));
                std::fill(array_ptr + i, array_ptr + i + l, 0);
            }
        }
        // Add bombs
        std::vector<std::pair<py::array_t<int>, std::pair<int, int>>> bombs_output = get_all_bombs_array(counts);
        outputs.insert(outputs.end(), bombs_output.begin(), bombs_output.end());
    }

    // 2 sequence
    else if (input_info == 10) {
        int l = input_leng / 2;
        for (int i = input_rank + 1; i < 15 - 2 - l; ++i) {
            bool invalid = false;
            for (int j = i; j < i + l; ++j) {
                if (counts[j] < 2) {
                    invalid = true;
                }
            }
            if (!invalid) {
                std::fill(array_ptr + i, array_ptr + i + l, 2);
                py::array_t<int> copy_arr = py::array_t<int>({size}, array_ptr);
                outputs.emplace_back(copy_arr, std::make_pair(10, i));
                std::fill(array_ptr + i, array_ptr + i + l, 0);
            }
        }
        // Add bombs
        std::vector<std::pair<py::array_t<int>, std::pair<int, int>>> bombs_output = get_all_bombs_array(counts);
        outputs.insert(outputs.end(), bombs_output.begin(), bombs_output.end());
    }

    // 3 sequence
    else if (input_info == 11) {
        int l = input_leng / 3;
        for (int i = input_rank + 1; i < 15 - 2 - l; ++i) {
            bool invalid = false;
            for (int j = i; j < i + l; ++j) {
                if (counts[j] < 3) {
                    invalid = true;
                }
            }
            if (!invalid) {
                std::fill(array_ptr + i, array_ptr + i + l, 3);
                py::array_t<int> copy_arr = py::array_t<int>({size}, array_ptr);
                outputs.emplace_back(copy_arr, std::make_pair(11, i));
                std::fill(array_ptr + i, array_ptr + i + l, 0);
            }
        }
        // Add bombs
        std::vector<std::pair<py::array_t<int>, std::pair<int, int>>> bombs_output = get_all_bombs_array(counts);
        outputs.insert(outputs.end(), bombs_output.begin(), bombs_output.end());
    }

    return outputs;
}


PYBIND11_MODULE(search_action_space, m) {
    m.def("get_all_actions", &get_all_actions, "equivalent to all_action python function except opinfo > 11. string output");
    m.def("get_avail_actions", &get_avail_actions, "equivalent to avail_action python function except opinfo > 11. string output");
    m.def("get_all_actions_array", &get_all_actions_array, "equivalent to all_action python function except opinfo > 11. array output");
    m.def("get_avail_actions_array", &get_avail_actions_array, "equivalent to avail_action python function except opinfo > 11. array output");
}
