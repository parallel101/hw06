#include <atomic>
#include <cstddef>
#include <functional>
#include <iostream>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <mutex>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_reduce.h>
#include <oneapi/tbb/parallel_scan.h>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "ticktock.h"

// TODO: 并行化所有这些 for 循环

template <class T, class Func>
std::vector<T> fill(std::vector<T> &arr, Func const &func) {
    TICK(fill);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, arr.size()), [&](tbb::blocked_range<size_t> r) {
        for (auto i = r.begin(); i != r.end(); i++) {
            arr[i] = func(i);
        }
    });
    TOCK(fill);
    return arr;
}

template <class T>
void saxpy(T a, std::vector<T> &x, std::vector<T> const &y) {
    TICK(saxpy);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, x.size()), [&](tbb::blocked_range<size_t> r) {
        for (auto i = r.begin(); i != r.end(); i++) {
            x[i] = a * x[i] + y[i];
        }
    });
    TOCK(saxpy);
}

template <class T>
T sqrtdot(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(sqrtdot);
    T ret = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, std::min(x.size(), y.size())),
        T{},
        [&](tbb::blocked_range<size_t> r, T local_res) {
            for (auto i = r.begin(); i != r.end(); i++) {
                local_res += x[i] * y[i];
            }
            return local_res;
        }, std::plus<T>{});
    ret = std::sqrt(ret);
    TOCK(sqrtdot);
    return ret;
}

template <class T>
T minvalue(std::vector<T> const &x) {
    TICK(minvalue);
    T ret = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, x.size()),
        std::numeric_limits<float>::max(),
        [&] (tbb::blocked_range<size_t> r, T local_min) {
            for (size_t i = r.begin(); i != r.end(); i++) {
                if (x[i] < local_min)
                    local_min = x[i];
            }
            return local_min;
        },
        [] (T x, T y) {
            return std::min(x,y);
        });
    TOCK(minvalue);
    return ret;
}

template <class T>
std::vector<T> magicfilter(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(magicfilter);
    auto minsize = std::min(x.size(), y.size());
    std::vector<T> res;
    std::mutex mtx;
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, minsize),
        [&] (tbb::blocked_range<size_t> r) {
            std::vector<T> local_vec;
            for (size_t i = r.begin(); i != r.end(); i++) {
                if (x[i] > y[i]) {
                    local_vec.push_back(x[i]);
                } else if (y[i] > x[i] && y[i] > 0.5f) {
                    local_vec.push_back(y[i]);
                    local_vec.push_back(x[i] * y[i]);
                }
            }
            std::unique_lock grd(mtx);
            std::copy(local_vec.begin(), local_vec.end(), std::back_inserter(res));
        });
    TOCK(magicfilter);
    return res;
}

template <class T>
T scanner(std::vector<T> &x) {
    TICK(scanner);
    T res = tbb::parallel_scan(
        tbb::blocked_range<size_t>(0, x.size()),
        T{},
        [&] (tbb::blocked_range<size_t> r, T local_res, auto is_final) {
            for (auto i = r.begin(); i != r.end(); i++) {
                local_res += x[i];
                if (is_final) 
                    x[i] = local_res;
            }
            return local_res;
        }, std::plus<T>());
    TOCK(scanner);
    return res;
}

int main() {
    size_t n = 1<<26;
    std::vector<float> x(n);
    std::vector<float> y(n);

    fill(x, [&] (size_t i) { return std::sin(i); });
    fill(y, [&] (size_t i) { return std::cos(i); });

    saxpy(0.5f, x, y);

    std::cout << sqrtdot(x, y) << std::endl;
    std::cout << minvalue(x) << std::endl;

    auto arr = magicfilter(x, y);
    std::cout << arr.size() << std::endl;

    scanner(x);
    std::cout << std::reduce(x.begin(), x.end()) << std::endl;

    return 0;
}
