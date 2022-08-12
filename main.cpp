#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <thread>
#include "ticktock.h"
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
#include <mutex>
#include <execution>

// TODO: 并行化所有这些 for 循环

template <class T, class Func>
std::vector<T> fill(std::vector<T> &arr, Func const &func) {
    TICK(fill);
    // ----------------------parallel_for---------------------------    
    tbb::parallel_for(tbb::blocked_range<size_t>(0, arr.size()),
        [&](auto r){
            for (size_t i = r.begin(); i < r.end(); i++) {
                arr[i] = func(i);
            }            
        }, tbb::auto_partitioner{});  

    // --------------------------old---------------------------
    // for (size_t i = 0; i < arr.size(); i++) {
    //     arr[i] = func(i);
    // }
    TOCK(fill);
    return arr;
}

template <class T>
void saxpy(T a, std::vector<T> &x, std::vector<T> const &y) {
    TICK(saxpy);
    // --------------------------parallel_for---------------------------
    tbb::parallel_for(tbb::blocked_range<size_t>(0, x.size()),
        [&](auto r){
            for (size_t i = r.begin(); i < r.end(); ++i){
                x[i] = a * x[i] + y[i];
            }
        });

    // --------------------------old---------------------------
    // for (size_t i = 0; i < x.size(); i++) {
    //    x[i] = a * x[i] + y[i];
    // }
    TOCK(saxpy);
}

template <class T>
T sqrtdot(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(sqrtdot);
    // --------------------------parallel_reduce---------------------------    
    T ret = std::sqrt(tbb::parallel_reduce(tbb::blocked_range<size_t>(0, std::min(x.size(), y.size())), (T)0, 
        [&](tbb::blocked_range<size_t> r, T local_res){
            for(size_t i=r.begin(); i<r.end(); ++i){
                local_res += x[i] * y[i];
            }
            return local_res;
        }, [](T x, T y){
            return x + y;
        }));

    // --------------------------old---------------------------    
    // T ret = 0;
    // for (size_t i = 0; i < std::min(x.size(), y.size()); i++) {
    //     ret += x[i] * y[i];
    // }
    // ret = std::sqrt(ret);
    TOCK(sqrtdot);
    return ret;
}

template <class T>
T minvalue(std::vector<T> const &x) {
    TICK(minvalue);
    // --------------------------parallel_for with mutex---------------------------     
    T ret = x[0];
    std::mutex mtx;    
    tbb::parallel_for(tbb::blocked_range<size_t>(1, x.size()),
        [&](auto r){
            T tmp = x[r.begin()];
            for(size_t i=r.begin()+1; i<r.end(); ++i)
                if(x[i] < tmp)
                    tmp = x[i];
            std::lock_guard lck(mtx);
            ret = std::min(ret, tmp);
        });

    // --------------------------parallel_reduce version---------------------------  
    // T ret = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, x.size()), (T)0,
    //     [&](tbb::blocked_range<size_t> r, T local_min){
    //         local_min = x[r.begin()];
    //         for(size_t i=r.begin()+1; i<r.end(); ++i){
    //             if(local_min > x[i])
    //                 local_min = x[i];
    //         }
    //         return local_min;
    //     }, [](T x, T y){
    //         return (x < y ? x : y);
    //     });

    // --------------------------old version---------------------------         
    // for (size_t i = 1; i < x.size(); i++) {
    //     if (x[i] < ret)
    //         ret = x[i];
    // }
    TOCK(minvalue);
    return ret;
}

template <class T>
std::vector<T> magicfilter(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(magicfilter);
    // --------------------------parallel_for with mutex--------------------------- 
    std::vector<T> res;
    size_t n = std::min(x.size(), y.size());
    res.reserve(n);
    std::mutex mtx;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](auto r){
        std::vector<T> tmp_arr;
        tmp_arr.reserve(r.size());
        for(size_t i=r.begin(); i<r.end(); ++i){
            if (x[i] > y[i]) {
                tmp_arr.push_back(x[i]);
            } else if (y[i] > x[i] && y[i] > 0.5f) {
                tmp_arr.push_back(y[i]);
                tmp_arr.push_back(x[i] * y[i]);
            }            
        }
        std::lock_guard lck(mtx);
        std::copy(tmp_arr.begin(), tmp_arr.end(), std::back_inserter(res));
    });

    // --------------------------old--------------------------- 
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
    // --------------------------parallel_scan--------------------------- 
    T ret = tbb::parallel_scan(tbb::blocked_range<size_t>(0, x.size()), (T)0, 
        [&](auto r, T local_res, auto is_final){
            for(size_t i=r.begin(); i<r.end(); ++i){
                local_res += x[i];
                if(is_final)
                    x[i] = local_res;
            }
            return local_res;
        }, [](T x, T y){
            return x + y;
        });

    // --------------------------old---------------------------
    // T ret = 0;
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
    std::cout << std::reduce(std::execution::seq, x.begin(), x.end()) << std::endl;

    return 0;
}
