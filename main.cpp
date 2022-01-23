#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
#include <mutex>
#include "ticktock.h"

// TODO: 并行化所有这些 for 循环

template <class T, class Func>
std::vector<T> fill(std::vector<T> &arr, Func const &func) {
    TICK(fill);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, arr.size()),
        [&](tbb::blocked_range<size_t> r) {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                arr[i] = func(i);
            }
        });
    TOCK(fill);
    return arr;
}

template <class T>
void saxpy(T a, std::vector<T> &x, std::vector<T> const &y) {
    TICK(saxpy);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, x.size()), 
        [&](tbb::blocked_range<size_t> r) {
            for ( size_t i = r.begin(); i < r.end(); ++i) {
                x[i] = a * x[i] + y[i];
            }
        });
    TOCK(saxpy);
}

template <class T>
T sqrtdot(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(sqrtdot);
    T ret = 0;
    ret = tbb::parallel_deterministic_reduce(
        tbb::blocked_range<size_t>(0, std::min(x.size(), y.size())),
        (T)0,
        [&](tbb::blocked_range<size_t> r, T local_res) {
            for ( size_t i = r.begin(); i < r.end(); ++i ) {
                local_res += x[i] * y[i];
            }
            return local_res;
        },
        []( T x, T y) {
            return x + y;
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
        [&] (tbb::blocked_range<size_t> r, T local_res) {
            for ( size_t i = r.begin(); i < r.end(); ++i) {
                if ( x[i] < local_res) {
                    local_res = x[i];
                }
            }
            return local_res;
        },
        [&] (T x, T y) {
            return x < y ? x : y;
        }
    );
    TOCK(minvalue);
    return ret;
}

template <class T>
std::vector<T> magicfilter(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(magicfilter);
    std::vector<T> res;
    std::mutex mtx;
    res.reserve( 2 * std::min(x.size(), y.size()));
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, std::min(x.size(), y.size())),
        [&] (tbb::blocked_range<size_t> r) {
            std::vector<T> tmp_vec;
            for (size_t i = r.begin(); i < r.end(); i++) {
                if (x[i] > y[i]) {
                    tmp_vec.push_back(x[i]);
                } else if (y[i] > x[i] && y[i] > 0.5f) {
                    tmp_vec.push_back(y[i]);
                    tmp_vec.push_back(x[i] * y[i]);
                }
            }
            
            std::lock_guard lck{mtx};
            std::copy(tmp_vec.begin(), tmp_vec.end(), std::back_inserter(res));
        }
    );
    TOCK(magicfilter);
    return res;
}

template <class T>
T scanner(std::vector<T> &x) {
    TICK(scanner);
    T ret = 0;
    // for (size_t i = 0; i < x.size(); i++) {
    //     ret += x[i];
    //     x[i] = ret;
    // }
    ret = tbb::parallel_scan(
        tbb::blocked_range<size_t>(0, x.size()),
        (T)0,
        [&] (tbb::blocked_range<size_t> r, T local_res, auto is_final) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                local_res += x[i];
                if (is_final) x[i] = local_res;
            }
            return local_res;
        },
        [] (T x, T y) {
            return  x + y;
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
// Before
    // fill: 1.23536s
    // fill: 1.25023s
    // saxpy: 0.048762s
    // sqrtdot: 0.086141s
    // 5165.4
    // minvalue: 0.07746s
    // -1.11803
    // magicfilter: 0.426949s
    // 55924034
    // scanner: 0.080046s
    // 5.28566e+07

// After Modified
    // fill: 0.256878s
    // fill: 0.264094s
    // saxpy: 0.03981s
    // sqrtdot: 1.18658s
    // 5792.62
    // minvalue: 0.01506s
    // -1.11803
    // magicfilter: 0.183792s
    // 55924034
    // scanner: 0.032603s
    // 5.28591e+07