#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <mutex>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
#include "ticktock.h"

// TODO: 并行化所有这些 for 循环

template <class T, class Func>
std::vector<T> fill(std::vector<T> &arr, Func const &func) {
    TICK(fill);
    tbb::parallel_for((size_t)0, arr.size(),
        [&](size_t i) {
            arr[i] = func(i);
        }
    );
    TOCK(fill);
    return arr;
}

template <class T>
void saxpy(T a, std::vector<T> &x, std::vector<T> const &y) {
    TICK(saxpy);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, x.size()),
        [&](tbb::blocked_range<size_t> r) {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                x[i] = a * x[i] + y[i];
            }
        }
    );
    TOCK(saxpy);
}

template <class T>
T sqrtdot(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(sqrtdot);
    size_t n = std::min(x.size(), y.size());
    T ret = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, n), T{},
        [&](tbb::blocked_range<size_t> r, T local_ret) {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                local_ret += x[i] * y[i];
            }
            return local_ret;
        },
        [](T x, T y) {
            return x + y;
        }
    );
    ret = std::sqrt(ret);
    TOCK(sqrtdot);
    return ret;
}

template <class T>
T minvalue(std::vector<T> const &x) {
    TICK(minvalue);
    T ret = tbb::parallel_reduce(tbb::blocked_range<size_t>(1, x.size()), x[0],
        [&](tbb::blocked_range<size_t> r, T local_ret) {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                if (x[i] < local_ret) {
                    local_ret = x[i];
                }
            }
            return local_ret;
        },
        [](T x, T y) {
            return std::min(x, y);
        }
    );
    TOCK(minvalue);
    return ret;
}

template <class T>
std::vector<T> magicfilter(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(magicfilter);
    size_t n = std::min(x.size(), y.size());

    std::vector<T> res;
    res.reserve(n);
    
    std::mutex mtx;

    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
        [&](tbb::blocked_range<size_t> r) {
            std::vector<T> tmp;
            tmp.reserve(r.size());
            for (size_t i = r.begin(); i < r.end(); ++i) {
                if (x[i] > y[i]) {
                    tmp.push_back(x[i]);
                }
                else if (y[i] > x[i] && y[i] > 0.5f) {
                    tmp.push_back(y[i]);
                    tmp.push_back(x[i] * y[i]);
                }
            }
            std::lock_guard  lck(mtx);
            std::copy(tmp.begin(), tmp.end(), std::back_inserter(res));
        }
    );
    TOCK(magicfilter);
    return res;
}

template <class T>
T scanner(std::vector<T> &x) {
    TICK(scanner);
    T ret = tbb::parallel_scan(tbb::blocked_range<size_t>(0, x.size()), T{},
        [&](tbb::blocked_range<size_t> r, T local_ret, auto is_final) {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                local_ret += x[i];
                if (is_final) {
                    x[i] = local_ret;
                }
            }
            return local_ret;
        },
        [](T x, T y) {
            return x + y;
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
