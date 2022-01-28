#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "ticktock.h"
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
#include <tbb/concurrent_vector.h>


// TODO: 并行化所有这些 for 循环

template <class T, class Func>
std::vector<T> fill(std::vector<T> &arr, Func const &func) {
    TICK(fill);
    const size_t n = arr.size();
    tbb::parallel_for((size_t)0, (size_t)n,[&](size_t i){
        arr[i] = func(i);
    });
    TOCK(fill);
    return arr;
}

template <class T>
void saxpy(T a, std::vector<T> &x, std::vector<T> const &y) {
    TICK(saxpy);
    const size_t n = x.size();
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
                      [&](tbb::blocked_range<size_t> r){
                          for(size_t i = r.begin(); i < r.end(); i++){
                              x[i] =  x[i]*a + y[i];
                          }
                      });
    TOCK(saxpy);
}

template <class T>
T sqrtdot(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(sqrtdot);
    T ret = 0;
    const size_t n = min(x.size(),y.size());
    ret = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, n),(T) 0,
                               [&](tbb::blocked_range<size_t> r, T local_ret){
                                   for(size_t i = r.begin(); i < r.end(); i++){
                                       local_ret += x[i]*y[i];
                                   }
                                   return local_ret;
                               },[](T x, T y){
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
    const size_t n = x.size();
//     tbb::parallel_for(tbb::blocked_range<size_t>(0,n),
//                       [&](tbb::blocked_range<size_t> r){
//                           for(size_t i = r.begin() ; i < r.end() ; ++i){
//                               if(x[i] < ret){
//                                   ret = x[i];
//                               }
//                           }
//                       });
    tbb::parallel_reduce(tbb::blocked_range<size_t>(0, n),(T) x[0],
                         [&](tbb::blocked_range<size_t> r, T local_min){
                             for(size_t i = r.begin(); i < r.end() ; i++){
                                 local_min = min(local_min,x[i]);
                             }
                             return local_min;},
                         [](T x, T y){
                             return min(x,y);
                         });
    TOCK(minvalue);
    return ret;
}

template <class T>
std::vector<T> magicfilter(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(magicfilter);
    std::vector<T> res;
    std::mutex mtx;
    const size_t n = min(x.size(),y.size());
    res.reserve(n);
    tbb::task_arena ta(6);
    ta.execute([&]{
        tbb::parallel_for(tbb::blocked_range<size_t>(0,n),
                          [&](tbb::blocked_range<size_t> r){
                              std::vector<T> local_res;
                              local_res.reserve(r.size());
                              for(size_t i = r.begin(); i < r.end(); ++i ){
                                  if(x[i] > y[i]){
                                      local_res.push_back(x[i]);
                                  }else if(y[i] > x[i] && y[i] > 0.5f){
                                      local_res.push_back(y[i]);
                                      local_res.push_back(x[i] * y[i]);
                                  }
                              }
                              std::lock_guard lck(mtx);
                              std::copy(local_res.begin(), local_res.end(), std::back_inserter(res));
                          },tbb::auto_partitioner{});
    });
    TOCK(magicfilter);
    return res;
}

template <class T>
T scanner(std::vector<T> &x) {
    TICK(scanner);
    T ret = 0;
    const size_t  n = x.size();
    ret = tbb::parallel_scan(tbb::blocked_range<size_t>(0,n),(T) 0 ,
                             [&](tbb::blocked_range<size_t> r, T local_ret, auto is_final){
                                 for(size_t i = r.begin(); i < r.end(); ++i){
                                     local_ret += x[i];
                                     if(is_final){
                                         x[i] = local_ret;
                                     }
                                 }
                                 return local_ret;
                             },
                             []( T x , T y){
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

    return 0;
}


