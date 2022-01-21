#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
#include <tbb/task_arena.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <numeric>
#include <vector>

#include "ticktock.h"

// TODO: 并行化所有这些 for 循环

template <class T, class Func>
std::vector<T> fill(std::vector<T> &arr, Func const &func) {
    TICK(fill);
    // 串行: 0.4s
    // for (size_t i = 0; i < arr.size(); i++) {
    //     arr[i] = func(i);
    // }

    // 并行映射，直接使用 parallel_for: 0.1s  加速比: 4x
    tbb::task_arena ta(4);
    ta.execute([&] {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, arr.size()), [&](tbb::blocked_range<size_t> r) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                arr[i] = func(i);
            }
        });
    });

    TOCK(fill);
    return arr;
}

template <class T>
void saxpy(T a, std::vector<T> &x, std::vector<T> const &y) {
    TICK(saxpy);
    // 串行: 0.015s
    // for (size_t i = 0; i < x.size(); i++) {
    //     x[i] = a * x[i] + y[i];
    // }

    // 并行: 0.015s, 加速比: 1x?
    tbb::task_arena ta(4);
    ta.execute([&] {
        auto num_procs = tbb::this_task_arena::max_concurrency();
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, x.size(), x.size() / (2 * num_procs)), [&](tbb::blocked_range<size_t> r) {
                for (size_t i = r.begin(); i < r.end(); i++) {
                    x[i] = a * x[i] + y[i];
                }
            },
            tbb::simple_partitioner{});
    });

    TOCK(saxpy);
}

template <class T>
T sqrtdot(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(sqrtdot);
    // 串行: 0.06s, output: 5165.4
    // T ret = 0;
    // for (size_t i = 0; i < std::min(x.size(), y.size()); i++) {
    //     ret += x[i] * y[i];
    // }
    // ret = std::sqrt(ret);

    // 并行 Reduce: 0.015s, 加速比: 4x, output:5792.63
    tbb::task_arena ta(4);
    T ret = 0;
    ta.execute([&] {
        ret = tbb::parallel_reduce(
            tbb::blocked_range<size_t>(0, std::min(x.size(), y.size())), (T)0, [&](tbb::blocked_range<size_t> r, T local_ret) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                local_ret += x[i] * y[i];
            }
            return local_ret; }, [](T x, T y) { return x + y; });
    });
    ret = std::sqrt(ret);
    TOCK(sqrtdot);
    return ret;
}

template <class T>
T minvalue(std::vector<T> const &x) {
    TICK(minvalue);
    // 串行: 0.08s, output:-1.11803
    // T ret = x[0];
    // for (size_t i = 1; i < x.size(); i++) {
    //     if (x[i] < ret)
    //         ret = x[i];
    // }
    // 并行 Reduce: 0.02s, 加速比: 4x, output:-1.11803
    tbb::task_arena ta(4);
    T ret = 0;
    ta.execute([&] {
        ret = tbb::parallel_reduce(
            tbb::blocked_range<size_t>(0, x.size()), (T)x[0], [&](tbb::blocked_range<size_t> r, T local_ret) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                if (x[i] < local_ret)
                    local_ret = x[i];
            }
            return local_ret; }, [](T x, T y) { return std::min(x, y); });
    });

    TOCK(minvalue);
    return ret;
}

template <class T>
std::vector<T> magicfilter(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(magicfilter);
    // 串行: 0.2s, output: 55924034
    std::vector<T> res;
    // for (size_t i = 0; i < std::min(x.size(), y.size()); i++) {
    //     if (x[i] > y[i]) {
    //         res.push_back(x[i]);
    //     } else if (y[i] > x[i] && y[i] > 0.5f) {
    //         res.push_back(y[i]);
    //         res.push_back(x[i] * y[i]);
    //     }
    // }
    // 并行 Filter: 0.14s, 加速比: 1.4x, output:55924034
    const size_t n = std::min(x.size(), y.size());
    res.reserve(n);
    std::mutex mtx;
    tbb::task_arena ta(4);
    ta.execute([&] {
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, n), [&](tbb::blocked_range<size_t> r) {
                std::vector<T> res_local;
                res_local.reserve(r.size());
                for (size_t i = r.begin(); i < r.end(); i++) {
                    if (x[i] > y[i]) {
                        res_local.push_back(x[i]);
                    } else if (y[i] > x[i] && y[i] > 0.5f) {
                        res_local.push_back(y[i]);
                        res_local.push_back(x[i] * y[i]);
                    }
                }
                std::lock_guard grd(mtx);
                std::copy(res_local.begin(), res_local.end(), std::back_inserter(res));
            },
            tbb::auto_partitioner{});
    });
    TOCK(magicfilter);
    return res;
}

template <class T>
T scanner(std::vector<T> &x) {
    TICK(scanner);
    T ret = 0;
    // 串行: 0.065s, output: 5.28566e+07
    // for (size_t i = 0; i < x.size(); i++) {
    //     ret += x[i];
    //     x[i] = ret;
    // }

    // 并行 Scan: 0.0354s, 加速比: 1.84x, output: 5.28685e+07
    tbb::task_arena ta(4);
    const size_t n = x.size();
    ta.execute([&] {
        ret = tbb::parallel_scan(
            tbb::blocked_range<size_t>(0, n), (T)0, [&](tbb::blocked_range<size_t> r, T local_res, auto is_final) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                local_res += x[i];
                if (is_final) {
                    x[i] = local_res;
                }
            }
            return local_res; }, [](float x, float y) { return x + y; });
    });

    TOCK(scanner);
    return ret;
}

int main() {
    size_t n = 1 << 26;
    std::vector<float> x(n);
    std::vector<float> y(n);

    fill(x, [&](size_t i) { return std::sin(i); });
    fill(y, [&](size_t i) { return std::cos(i); });

    saxpy(0.5f, x, y);

    std::cout << sqrtdot(x, y) << std::endl;
    std::cout << minvalue(x) << std::endl;

    auto arr = magicfilter(x, y);
    std::cout << arr.size() << std::endl;

    scanner(x);
    std::cout << std::reduce(x.begin(), x.end()) << std::endl;

    return 0;
}
