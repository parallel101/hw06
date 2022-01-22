#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <tbb/tbb.h>
#include <mutex>
#include <tbb/spin_mutex.h>

#include "ticktock.h"

// TODO: 并行化所有这些 for 循环

template <class T, class Func>
std::vector<T> fill(std::vector<T> &arr, Func const &func) {
    TICK(fill);
    for (size_t i = 0; i < arr.size(); i++) {
        arr[i] = func(i);
    }
    TOCK(fill);

    return arr;
}

template <class T, class Func>
std::vector<T> fill_parallel(std::vector<T> &arr, Func const &func) {
    TICK(fill_parallel);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, arr.size()), [&](tbb::blocked_range<size_t> r) {
        for (size_t i = r.begin(); i < r.end(); i++) {
            arr[i] = func(i);
        }
    });
    TOCK(fill_parallel);

    return arr;
}

template <class T>
void saxpy(T a, std::vector<T> &x, std::vector<T> const &y) {
    TICK(saxpy);
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = a * x[i] + y[i];
    }
    TOCK(saxpy);
}

template <class T>
void saxpy_parallel(T a, std::vector<T> &x, std::vector<T> const &y) {
    TICK(saxpy_parallel);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, x.size()), [&](tbb::blocked_range<size_t> r) {
        for (size_t i = r.begin(); i < r.end(); i++) {
            x[i] = a * x[i] + y[i];
        }
    });
    TOCK(saxpy_parallel);
}

template <class T>
T sqrtdot(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(sqrtdot);
    T ret = 0;
    for (size_t i = 0; i < std::min(x.size(), y.size()); i++) {
        ret += x[i] * y[i];
    }
    ret = std::sqrt(ret);
    TOCK(sqrtdot);
    return ret;
}

template <class T>
T sqrtdot_parallel(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(sqrtdot_parallel);
    float ret = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, x.size()), (T)0,
        [&](tbb::blocked_range<size_t> r, float local_res) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                local_res += x[i] * y[i];
            }
            return local_res;
        },
        [](float x, float y) { return x + y; });
    ret = std::sqrt(ret);
    TOCK(sqrtdot_parallel);
    return ret;
}

template <class T>
T minvalue(std::vector<T> const &x) {
    TICK(minvalue);
    T ret = x[0];
    for (size_t i = 1; i < x.size(); i++) {
        if (x[i] < ret)
            ret = x[i];
    }
    TOCK(minvalue);
    return ret;
}

template <class T>
T minvalue_parallel(std::vector<T> const &x) {
    TICK(minvalue_parallel);
    float ret = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, x.size()), (T)0,
        [&](tbb::blocked_range<size_t> r, float local_res) {
            local_res = x[r.begin()];
            for (size_t i = r.begin(); i < r.end(); i++) {
                if (x[i] < local_res) {
                    local_res = x[i];
                }
            }
            return local_res;
        },
        [](float x, float y) {
            if (x < y) {
                return x;
            } else {
                return y;
            };
        });
    TOCK(minvalue_parallel);
    return ret;
}

template <class T>
std::vector<T> magicfilter(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(magicfilter);
    std::vector<T> res;
    for (size_t i = 0; i < std::min(x.size(), y.size()); i++) {
        if (x[i] > y[i]) {
            res.push_back(x[i]);
        } else if (y[i] > x[i] && y[i] > 0.5f) {
            res.push_back(y[i]);
            res.push_back(x[i] * y[i]);
        }
    }
    TOCK(magicfilter);
    return res;
}

template <class T>
std::vector<T> magicfilter_parallel(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(magicfilter_parallel);
    std::mutex mtx;
    // tbb::spin_mutex mtx;
    std::vector<T> res;
    size_t n = std::min(x.size(), y.size());
    res.reserve(n);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](tbb::blocked_range<size_t> r) {
        std::vector<T> local_a;
        local_a.reserve(r.size());
        for (size_t i = r.begin(); i < r.end(); i++) {
            if (x[i] > y[i]) {
                local_a.push_back(x[i]);
            } else if (y[i] > x[i] && y[i] > 0.5f) {
                local_a.push_back(y[i]);
                local_a.push_back(x[i] * y[i]);
            }
        }
        std::lock_guard grd{mtx};
        std::copy(local_a.begin(), local_a.end(), std::back_inserter(res));
    });
    TOCK(magicfilter_parallel);
    return res;
}

template <class T>
T scanner(std::vector<T> &x) {
    TICK(scanner);
    T ret = 0;
    for (size_t i = 0; i < x.size(); i++) {
        ret += x[i];
        x[i] = ret;
    }
    TOCK(scanner);
    return ret;
}

template <class T>
T scanner_patallel(std::vector<T> &x) {
    TICK(scanner_patallel);
    size_t n = x.size();
    auto ret = tbb::parallel_scan(
        tbb::blocked_range<size_t>(0, n), (T)0,
        [&](tbb::blocked_range<size_t> r, size_t local_res, auto is_final) {
            for (auto i = r.begin(); i < r.end(); i++) {
                local_res += x[i];
                if (is_final)
                    x[i] = local_res;
            }
            return local_res;
        },
        [](size_t x, size_t y) { return x + y; });
    TOCK(scanner_patallel);
    return ret;
}

int main() {
    size_t n = 1 << 26;
    std::vector<float> x(n);
    std::vector<float> y(n);
    std::vector<float> z(n);
    std::vector<float> w(n);

    fill(x, [&](size_t i) { return std::sin(i); });
    fill(y, [&](size_t i) { return std::cos(i); });

    fill_parallel(z, [&](size_t i) { return std::sin(i); });
    fill_parallel(w, [&](size_t i) { return std::cos(i); });

    saxpy(0.5f, x, y);
    saxpy_parallel(0.5f, z, w);

    std::cout << sqrtdot(x, y) << std::endl;
    std::cout << sqrtdot_parallel(x, y) << std::endl;

    std::cout << minvalue(x) << std::endl;
    std::cout << minvalue_parallel(x) << std::endl;

    auto arr = magicfilter(x, y);
    std::cout << arr.size() << std::endl;
    auto arr_p = magicfilter_parallel(x, y);
    std::cout << arr_p.size() << std::endl;

    scanner(x);
    std::cout << std::reduce(x.begin(), x.end()) << std::endl;
    auto k = x;
    scanner_patallel(y);
    std::cout << std::reduce(k.begin(), k.end()) << std::endl;

    return 0;
}
