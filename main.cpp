#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "ticktock.h"

#include <mutex>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#include <tbb/spin_mutex.h>
#include <tbb/parallel_scan.h>
#include <atomic>
#include <memory>
#include <xmmintrin.h>

// TODO: 并行化所有这些 for 循环

template <class T, class Func>
std::vector<T> fill(std::vector<T> &arr, Func const &func) {
    TICK(fill);

    tbb::parallel_for((size_t)0, arr.size(), 
    [&] (size_t i) {
        arr[i] = func(i);
    });

    TOCK(fill);
    return arr;
}

template <class T>
void saxpy(T a, std::vector<T> __restrict &x, std::vector<T> const __restrict &y) {
    TICK(saxpy);
    tbb::task_arena ta(4);
    ta.execute([&] {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, x.size(), 4),
        [&] (tbb::blocked_range<size_t> r) {
            for (size_t i = r.begin(); i < r.end(); i+=r.grainsize()) {
                auto ma = _mm_set_ps1(a);
                auto mx = _mm_load_ps(&x[i]);
                auto my = _mm_load_ps(&y[i]);
                auto res = _mm_add_ps(_mm_mul_ps(ma, mx), my);
                _mm_store_ps(&x[i], res);
                // x[i] = a * x[i] + y[i];
            }
        });
    });

    TOCK(saxpy);
}

template <class T>
T sqrtdot(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(sqrtdot);

    size_t n = std::min(x.size(), y.size());
    std::atomic<float> aret = ATOMIC_VAR_INIT(0);
    
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n), 
    [&](tbb::blocked_range<size_t> r) {
        T local_xmy = 0;
#pragma omp simd
        for (size_t i = r.begin(); i < r.end(); i++) {
            local_xmy += x[i]*y[i];
        }
        T val = aret.load();
        while (!aret.compare_exchange_strong(val, val+local_xmy));
    });

    T aaret = std::sqrt(aret.load());
    TOCK(sqrtdot);
    return aaret;
}

template <class T>
T minvalue(std::vector<T> const &x) {
    TICK(minvalue);
    std::atomic<T> atm = ATOMIC_VAR_INIT(x[0]);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, x.size()),
    [&] (tbb::blocked_range<size_t> r) {
        T local_min_value = x[r.begin()];
        for (size_t i = r.begin() + 1; i < r.end(); i++) {
            if (x[i] < local_min_value)
                local_min_value = x[i];
        }
        
        T old = atm.load();
        while (local_min_value < old && !atm.compare_exchange_weak(old, local_min_value));
    }, tbb::auto_partitioner{});

    T ret = atm.load();

    TOCK(minvalue);
    return ret;
}

template <class T>
std::vector<T> magicfilter(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(magicfilter);

    size_t n = std::min(x.size(), y.size());
    std::mutex mtx;

    std::vector<T> res;
    res.reserve(n*3);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n), 
    [&](tbb::blocked_range<size_t> r) {
        static thread_local std::vector<T> local;
        local.clear();
        local.reserve(r.size()*2);
        for (size_t i = r.begin(); i < r.end(); i++) {
            if (x[i] > y[i]) {
                local.push_back(x[i]);
            }
            else if (y[i] > 0.5f && y[i] > x[i]) {
                local.push_back(y[i]);
                local.push_back(x[i]*y[i]);
            }
        }
        std::lock_guard lck(mtx);
        std::copy(local.begin(), local.end(), std::back_inserter(res));
    });
    TOCK(magicfilter);
    return res;
}

template <class T>
T scanner(std::vector<T> &x) {
    TICK(scanner);
    T ret = 0;

    ret  = tbb::parallel_scan(tbb::blocked_range<size_t>(0, x.size()), (float)0,
    [&] (tbb::blocked_range<size_t> r, T sum, auto is_final_scan)->T {
        T temp = sum;
        for (size_t i = r.begin(); i < r.end(); i++) {
            temp += x[i];
            if (is_final_scan) {
                x[i] = temp;
            }
        }
        return temp;
    }, 
    [] (T left, T right) {
        return left + right;
    });

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