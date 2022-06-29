#include "mtprint.h"
#include "ticktock.h"
#include "pod.h"
#include <algorithm>
#include <cmath>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
#include <vector>
#include <mutex>


template <class T, class Func>
std::vector<T> fill(std::vector<T> &arr, Func const &func) {
    TICK(fill);
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, arr.size()),
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
    const size_t s = x.size();
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, x.size()),
        [&](tbb::blocked_range<size_t> r) {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                x[i] = a * x[i] + y[i];
            }
        });
    TOCK(saxpy);
}

template <class T> T sqrtdot(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(sqrtdot);
    T ret = 0;
    size_t n = std::min(x.size(), y.size());

    T res = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, n), (T)0,
        [&](tbb::blocked_range<size_t> r, T local_res) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                local_res += x[i] * y[i];
            }
            return local_res;
        },
        [](T x, T y) { return x + y; });

    ret = std::sqrt(ret);
    TOCK(sqrtdot);
    return ret;
}

template <class T> T minvalue(std::vector<T> const &x) {
    TICK(minvalue);
    T ret = x[0];
    T res = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, x.size()), (T)0,
        [&](tbb::blocked_range<size_t> r, T local_res) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                local_res = std::min(x[i], local_res);
            }
            return local_res;
        },
        [](T x, T y) { return std::min(x, y); });
    TOCK(minvalue);
    return ret;
}

template <class T>
std::vector<T> magicfilter(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(magicfilter);
    std::vector<T> res(std::min(x.size(), y.size()) * 17 / 20);
    std::atomic<size_t> rsize = 0;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, std::min(x.size(), y.size())),
    [&] (tbb::blocked_range<size_t> r) {
        size_t lsize = 0;
        std::vector<pod<T>> local_res(r.size() * 2);
        for (size_t i = r.begin(); i < r.end(); i++) {
            if (x[i] > y[i]) {
                local_res[lsize++] = x[i];
            } else if (y[i] > x[i] && y[i] > 0.5f) {
                local_res[lsize++] = y[i];
                local_res[lsize++] = x[i] * y[i];
            }
        }

        size_t base = rsize.fetch_add(lsize);
        for (size_t i=0; i<lsize; ++i)
            res[base + i] = local_res[i];
    });

    TOCK(magicfilter);
    return res;
}

template <class T> T scanner(std::vector<T> &x) {
    TICK(scanner);
    T ret = 0;
    for (size_t i = 0; i < x.size(); i++) {
        ret += x[i];
        x[i] = ret;
    }

    T res = tbb::parallel_scan(tbb::blocked_range<size_t>(0, x.size()), (T)0,
    [&] (tbb::blocked_range<size_t> r, T local_res, auto is_final) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
            local_res += x[i];
            if (is_final) {
                x[i] = local_res;
            }
        }
        return local_res;
    }, [] (T x, T y) {
        return x + y;
    });
    TOCK(scanner);
    return ret;
}

int main() {
    size_t n = 1 << 26;
    std::vector<float> x(n);
    std::vector<float> y(n);

    fill(x, [&](size_t i) { return ::sinf(i); });
    fill(y, [&](size_t i) { return ::cosf(i); });

    saxpy(0.5f, x, y);

    std::cout << sqrtdot(x, y) << std::endl;
    std::cout << minvalue(x) << std::endl;

    auto arr = magicfilter(x, y);
    std::cout << arr.size() << std::endl;

    scanner(x);
    std::cout << std::reduce(x.begin(), x.end()) << std::endl;

    return 0;
}
