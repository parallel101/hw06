#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "ticktock.h"

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
#include <mutex>
#include "pod.h"

// TODO: 并行化所有这些 for 循环

// origin:
// fill: 0.713545s
// fill: 0.726485s
// saxpy: 0.0369504s
// sqrtdot: 0.0733301s
// 5165.4
// minvalue: 0.0715443s
// -1.11803
// magicfilter: 0.190107s
// 55924034
// scanner: 0.0710648s
// 6.18926e+07

// parallel:
// fill: 0.106531s
// fill: 0.109878s
// saxpy: 0.0345495s
// sqrtdot: 0.0193823s
// 5792.61
// minvalue: 0.0108345s
// -1.11803
// magicfilter: 0.126976s
// 55924034
// scanner: 0.0371496s
// 6.19079e+07

template <class T, class Func>
std::vector<T> fill(std::vector<T> &arr, Func const &func) {
    TICK(fill);
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, arr.size()),
        [&](tbb::blocked_range<size_t> r) {
            for (size_t i = r.begin(); i < r.end(); ++i)
                arr[i] = func(i);
        });
    TOCK(fill);
    return arr;
}

template <class T>
void saxpy(T a, std::vector<T> &x, std::vector<T> const &y) {
    TICK(saxpy);
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, x.size()),
        [&](tbb::blocked_range<size_t> r) {
            for (size_t i = r.begin(); i < r.end(); ++i)
                x[i] = a * x[i] + y[i];
        });
    TOCK(saxpy);
}

template <class T>
T sqrtdot(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(sqrtdot);
    T ret = 0;
    ret = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, std::min(x.size(), y.size())),
        static_cast<T>(0),
        [&](tbb::blocked_range<size_t> r, T temp_sum) {
            for (size_t i = r.begin(); i < r.end(); ++i)
                temp_sum += x[i] * y[i];
            return temp_sum;
        },
        [](T a, T b) {
            return a + b;
        });
    ret = std::sqrt(ret);
    TOCK(sqrtdot);
    return ret;
}

template <class T>
T minvalue(std::vector<T> const &x) {
    TICK(minvalue);
    T ret = x[0];
    ret = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(1, x.size()),
        x[0],
        [&](tbb::blocked_range<size_t> r, T temp_min) {
            for ( size_t i = r.begin(); i < r.end(); ++i)
                if (temp_min > x[i])
                    temp_min = x[i];
            return temp_min;
        },
        [](T a, T b) {
            return std::min(a, b);
        }
    );
    TOCK(minvalue);
    return ret; 
}

template <class T>
std::vector<pod<T>> magicfilter(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(magicfilter);
    std::vector<pod<T>> res;
    res.reserve(2 * std::min(x.size(), y.size()));
    std::mutex mtx;
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, std::min(x.size(), y.size())),
        [&](tbb::blocked_range<size_t> r) {
            std::vector<pod<T>> cache;
            cache.reserve(2 * r.size());
            for (size_t i = r.begin(); i < r.end(); ++i) {
                if (x[i] > y[i])
                    cache.push_back(x[i]);
                else if (y[i] > x[i] && y[i] > 0.5f) {
                    cache.push_back(y[i]);
                    cache.push_back(x[i] * y[i]);
                }
            }
            std::lock_guard lck(mtx);
            std::copy(cache.begin(), cache.end(), std::back_inserter(res));
        }
    );

    TOCK(magicfilter);
    return res;
}

template <class T>
T scanner(std::vector<T> &x) {
    TICK(scanner);
    T ret = 0;
    ret = tbb::parallel_scan(
        tbb::blocked_range<size_t>(0, x.size()),
        static_cast<T>(0),
        [&] (tbb::blocked_range<size_t> r, T temp_sum, auto is_final) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                temp_sum += x[i];
                if (is_final) 
                    x[i] = temp_sum;
            }
            return temp_sum;
        },
        [] (T a, T b) {
            return  a + b;
        }
    );
    TOCK(scanner);
    return ret;
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
