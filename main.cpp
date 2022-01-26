#include <cstddef>
#include <iostream>
#include <cstdlib>
#include <iterator>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_reduce.h>
#include <oneapi/tbb/parallel_scan.h>
#include <oneapi/tbb/partitioner.h>
#include <oneapi/tbb/spin_mutex.h>
#include <oneapi/tbb/task_arena.h>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <mutex>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
#include <tbb/spin_mutex.h>
#include "ticktock.h"

// TODO: 并行化所有这些 for 循环

template <class T, class Func>
std::vector<T> fill(std::vector<T> &arr, Func const &func) {
    TICK(fill);
    size_t len = arr.size();
    #pragma omp parallel for
    for (size_t i = 0; i < len; i++) {
        arr[i] = func(i);
    }
    TOCK(fill);
    return arr;
}

template <class T>
void saxpy(T a, std::vector<T> &x, std::vector<T> const &y) {
    TICK(saxpy);
    #pragma omp parallel for
    for (size_t i = 0; i < x.size(); i++) {
       x[i] = a * x[i] + y[i];
    }
    TOCK(saxpy);
}

template <class T>
T sqrtdot(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(sqrtdot);
    size_t len = std::min(x.size(), y.size());
    T ret = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, len), T{},
    [&](tbb::blocked_range<size_t> r, T local_res){
        for(size_t i = r.begin(); i<r.end(); i++){
            local_res += x[i] * y[i];
        }
        return local_res;
    }, [](T x, T y){
        return x + y;
    });


/*
    T ret = 0;
    for (size_t i = 0; i < std::min(x.size(), y.size()); i++) {
        ret += x[i] * y[i];
    }
*/
    ret = std::sqrt(ret);
    TOCK(sqrtdot);
    return ret;
}

template <class T>
T minvalue(std::vector<T> const &x) {
    TICK(minvalue);
    T ret = x[0];
    #pragma omp parallel for
    for (size_t i = 1; i < x.size(); i++) {
        if (x[i] < ret)
            ret = x[i];
    }
    TOCK(minvalue);
    return ret;
}

template <class T>
std::vector<T> magicfilter(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(magicfilter);
    std::vector<T> res;

    size_t n = std::min(x.size(), y.size());
    res.reserve(n*3);
    tbb::spin_mutex mtx;
    tbb::task_arena ta(20);
    ta.execute([&]{
            tbb::parallel_for(tbb::blocked_range<size_t>(0, n), 
            [&](tbb::blocked_range<size_t> r){
                std::vector<T> local_res;
                local_res.reserve(r.size()*2);
                for(size_t i=r.begin(); i<r.end(); i++){
                    if (x[i] > y[i]) {
                        local_res.push_back(x[i]);
                    } else if (y[i] > x[i] && y[i] > 0.5f) {
                        local_res.push_back(y[i]);
                        local_res.push_back(x[i] * y[i]);
                    }
                }
                std::lock_guard<tbb::spin_mutex> lock_guard(mtx);
                std::copy(local_res.begin(), local_res.end(), std::back_inserter(res));
            }, tbb::auto_partitioner{});
    }
    );
    

    // for (size_t i = 0; i < std::min(x.size(), y.size()); i++) {
    //     if (x[i] > y[i]) {
    //         res.push_back(x[i]);
    //     } else if (y[i] > x[i] && y[i] > 0.5f) {
    //         res.push_back(y[i]);
    //         res.push_back(x[i] * y[i]);
    //     }
    // }

    TOCK(magicfilter);
    return res;
}

template <class T>
T scanner(std::vector<T> &x) {
    TICK(scanner);
    T ret = 0;
    tbb::parallel_scan(tbb::blocked_range<size_t>(0, x.size()), T{}, 
        [&](tbb::blocked_range<size_t> r, T local_res, auto is_final){
            for(size_t i = r.begin(); i<r.end(); i++){
                local_res += x[i];
                if(is_final){
                    x[i] = local_res;
                }
            }
            return local_res;
    }, [](T x, T y){
        return x + y;
    });
    // for (size_t i = 0; i < x.size(); i++) {
    //     ret += x[i];
    //     x[i] = ret;
    // }
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
