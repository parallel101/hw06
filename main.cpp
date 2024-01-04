#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "ticktock.h"
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>

// TODO: 并行化所有这些 for 循环

template <class T, class Func>
std::vector<T> fill(std::vector<T> &arr, Func const &func) {
    TICK(fill);
    tbb::task_arena ta(8);
    ta.execute([&]{
        tbb::parallel_for(tbb::blocked_range<size_t>(0,arr.size()),[&](tbb::blocked_range<size_t> r){
            for (size_t i = r.begin(); i !=r.end(); i++) {
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
    tbb::task_arena ta(8);
    ta.execute([&]{
        tbb::parallel_for(tbb::blocked_range<size_t>(0,x.size()),[&](tbb::blocked_range<size_t> r){
            for (size_t i = r.begin(); i !=r.end(); i++) {
                x[i] = a * x[i] + y[i];
            }
        });
    });

    TOCK(saxpy);
}

template <class T>
T sqrtdot(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(sqrtdot);
    T ret = 0;
    tbb::task_arena ta(8);
    ta.execute([&]{
        ret=tbb::parallel_reduce(tbb::blocked_range<size_t>(0,std::min(x.size(), y.size())),(T )0,
             [&](tbb::blocked_range<size_t> r, T local_res){
                 for (size_t i = r.begin(); i != r.end(); i++) {
                     local_res += x[i] * y[i];
                 }
                 return local_res;
             },
             [](T x,T y){
                 return x+y;
             });
    });
    ret = std::sqrt(ret);
    TOCK(sqrtdot);
    return ret;
}

template <class T>
T minvalue(std::vector<T> const &x) {
    TICK(minvalue);
    T ret = x[0];
    tbb::task_arena ta(8);
    ta.execute([&]{
        ret=tbb::parallel_reduce(tbb::blocked_range<size_t>(0, x.size()),x[0],
             [&](tbb::blocked_range<size_t> r, T local_res){
                 for (size_t i = r.begin()+1; i != r.end(); i++) {
                     if (x[i] < local_res)
                         local_res = x[i];
                 }
                 return local_res;
             },
             [](T x,T y){
                 return std::min(x,y);
             });
    });

    TOCK(minvalue);
    return ret;
}

template <class T>
std::vector<T> magicfilter(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(magicfilter);
    std::vector<T> res;

    std::atomic<size_t> a_size=0;
    size_t n = std::min(x.size(), y.size());
    res.resize(n);

    tbb::task_arena ta(8);
    ta.execute([&]{
        tbb::parallel_for(tbb::blocked_range<size_t>(0,n),
              [&](tbb::blocked_range<size_t> r){
                  std::vector<T> la(r.size());
                  size_t la_idx=0;
                  for (size_t i = r.begin(); i < r.end(); i++) {
                      if (x[i] > y[i]) {
                          la[la_idx++]=x[i];
                      } else if (y[i] > x[i] && y[i] > 0.5f) {
                          la[la_idx++]=y[i];
                          la[la_idx++]=x[i] * y[i];
                      }
                  }

                  size_t base=a_size.fetch_add(la_idx);
                  for (size_t i = 0; i < la_idx; ++i) {
                      res[base+i]=la[i];
                  }
              });
    });

    res.resize(a_size);
    TOCK(magicfilter);
    return res;
}

template <class T>
T scanner(std::vector<T> &x) {
    TICK(scanner);
    T ret=0;
    tbb::task_arena ta(8);
    ta.execute([&]{
        ret = tbb::parallel_scan(tbb::blocked_range<size_t>(0,x.size()),(T)0,
           [&](tbb::blocked_range<size_t> r,T local_res,auto is_final){
               for (size_t i = r.begin(); i < r.end(); i++) {
                   local_res += x[i];
                   if(is_final){
                       x[i] = ret;
                   }
               }

               return local_res;
           },
           [](T x,T y){
               return x+y;
           });
    });


//    T ret=0;
//    for (size_t i = 0; i < x.size(); i++) {
//        ret += x[i];
//        x[i] = ret;
//    }
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
