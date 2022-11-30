#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "ticktock.h"
#include <mutex>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
// TODO: 并行化所有这些 for 循环

template <class T, class Func>
std::vector<T> fill(std::vector<T> &arr, Func const &func) {
    TICK(fill);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, arr.size()), 
    [&func, &arr](const tbb::blocked_range<size_t>& r) {
        for (auto i = r.begin(); i != r.end(); i++)
        {
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
    [a, &x, &y](const tbb::blocked_range<size_t>& r) {
        for (auto i = r.begin(); i != r.end(); i++)
        {
            x[i] = a * x[i] + y[i];
        }
    });
    TOCK(saxpy);
}

template <class T>
T sqrtdot(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(sqrtdot);
    T ret = 0;
    ret = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, std::min(x.size(), y.size())),
    T{},
    [&x, &y](const tbb::blocked_range<size_t>& r, T local_val) {
        for (size_t i = r.begin(); i != r.end(); i++)
        {
            local_val += x[i] * y[i];
        }
        return local_val;
    }, std::plus<T>{});
    ret = std::sqrt(ret);
    TOCK(sqrtdot);
    return ret;
}

template <class T>
T minvalue(std::vector<T> const &x) {
    TICK(minvalue);
    T ret = tbb::parallel_reduce(tbb::blocked_range<size_t>(1, x.size()),
    x[0],
    [&x](const tbb::blocked_range<size_t>& r, T local_val) {
        for (size_t i = r.begin(); i != r.end(); ++i) 
            if (x[i] < local_val)
                local_val = x[i];
        return local_val;
    }, 
    [](T x, T y) { return std::min(x, y);});
    TOCK(minvalue);
    return ret;
}

template <class T>
std::vector<T> magicfilter(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(magicfilter);
    std::vector<T> res;
    size_t MIN_LEN = std::min(x.size(), y.size());
    res.reserve( MIN_LEN * 2 / 3);
    std::mutex mut;

    tbb::parallel_for(tbb::blocked_range<size_t>(0, MIN_LEN),
    [&res, &mut, &x, &y](const tbb::blocked_range<size_t>& r) {
        std::vector<T> local_vec;
        local_vec.reserve(r.size());
        for (auto i = r.begin(); i != r.end(); i++)
            if (x[i] > y[i]) {
                local_vec.push_back(x[i]);
            } else if (y[i] > x[i] && y[i] > 0.5f) {
                local_vec.push_back(y[i]);
                local_vec.push_back(x[i] * y[i]);
            }
        std::lock_guard lck(mut);
        std::copy(local_vec.begin(), local_vec.end(), std::back_inserter(res));
    });
    TOCK(magicfilter);
    return res;
}

template <class T>
T scanner(std::vector<T> &x) {
    TICK(scanner);
    T ret = 0;
    ret = tbb::parallel_scan(tbb::blocked_range<size_t>(0, x.size()),
    T{},
    [&x](const tbb::blocked_range<size_t>& r, T local_res, bool is_final) {
        for (size_t i = r.begin(); i != r.end(); i++)
        {
            local_res += x[i];
            if (is_final)
                x[i] = local_res;    
        }
        return local_res;
    }, std::plus<T>{});
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
