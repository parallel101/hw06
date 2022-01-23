#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/task_group.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
#include "ticktock.h"
// TODO: 并行化所有这些 for 循环

template <class T, class Func>
std::vector<T> fill(std::vector<T> &arr, Func const &func) {
    TICK(fill);
    auto n = arr.size();
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
        [&](tbb::blocked_range<size_t> r) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                arr[i] = func(i);
            }
        });
    //for (size_t i = 0; i < arr.size(); i++) {
    //    arr[i] = func(i);
    //}
    TOCK(fill);
    return arr;
}

template <class T>
void saxpy(T a, std::vector<T> &x, std::vector<T> const &y) {
    TICK(saxpy);
    auto n = std::min(x.size(), y.size());
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
        [&](tbb::blocked_range<size_t> r) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                //arr[i] = func(i);
                x[i] = a * x[i] + y[i];
            }
        });
    //for (size_t i = 0; i < x.size(); i++) {
    //   x[i] = a * x[i] + y[i];
    //}
    TOCK(saxpy);
}

template <class T>
T sqrtdot(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(sqrtdot);
    T ret = 0;
    auto n = std::min(x.size(), y.size());
    tbb::task_arena ta(4);
    auto res = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, n), (T)0,
        [&](tbb::blocked_range<size_t> r, T local_res) {
            for (size_t i = r.begin(); i < r.end(); ++i)
            {
                local_res += x[i] * y[i];
            }
            return local_res;
        }, [](T x, T y)
        {
            return x + y;
        });

    //for (size_t i = 0; i < std::min(x.size(), y.size()); i++) {
    //    ret += x[i] * y[i];
    //}
    ret = std::sqrt(res);
    //ret = std::sqrt(ret);
    TOCK(sqrtdot);
    return ret;
}

template <class T>
T minvalue(std::vector<T> const &x) {
    TICK(minvalue);
    auto n = x.size();
    T ret = x[0];

    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
        [&](tbb::blocked_range<size_t> r) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                if (x[i] < ret)
                    ret = x[i];
            }
        });

    //for (size_t i = 1; i < x.size(); i++) {
    //    if (x[i] < ret)
    //        ret = x[i];
    //}
    TOCK(minvalue);
    return ret;
}

template <class T>
std::vector<T> magicfilter(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(magicfilter);
    std::vector<T> res;
    auto n = std::min(x.size(), y.size());
    res.reserve(n * 2 / 3);
    std::mutex mtx;

    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
        [&](tbb::blocked_range<size_t> r) {
            std::vector<T> local_a;
            local_a.reserve(n);
            for (size_t i = r.begin(); i < r.end(); ++i) {
                if (x[i] > y[i]) {
                    local_a.push_back(x[i]);
                }
                else if (y[i] > x[i] && y[i] > 0.5f) {
                    local_a.push_back(y[i]);
                    local_a.push_back(x[i] * y[i]);
                }
            }
            std::lock_guard grd(mtx);
            std::copy(local_a.begin(), local_a.end(), std::back_inserter(res));
        }
    );
    //for (size_t i = 0; i < std::min(x.size(), y.size()); i++) {
    //    if (x[i] > y[i]) {
    //        res.push_back(x[i]);
    //    } else if (y[i] > x[i] && y[i] > 0.5f) {
    //        res.push_back(y[i]);
    //        res.push_back(x[i] * y[i]);
    //    }
    //}
    TOCK(magicfilter);
    return res;
}

template <class T>
T scanner(std::vector<T> &x) {
    TICK(scanner);
    T ret = 0;
    //for (size_t i = 0; i < x.size(); i++) {
    //    ret += x[i];
    //    x[i] = ret;
    //}

    auto n = x.size();
    ret = tbb::parallel_scan(tbb::blocked_range<size_t>(0, n), (T)0,
        [&](tbb::blocked_range<size_t> r, T local_res, auto is_final) {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                local_res += x[i];
                if (is_final) {
                    x[i] = local_res;
                }
            }

            return local_res;
        }, [](float x, float y) {
            return x + y;
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

    std::getchar();

    return 0;
}
