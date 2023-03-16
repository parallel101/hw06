#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "ticktock.h"
#include "tbb/tbb.h"
#include <atomic>
#include "pod.h"

//显式定义线程数
#define NUM_THREADS 8
// TODO: 并行化所有这些 for 循环

template <class T, class Func>
std::vector<T> fill(std::vector<T> &arr, Func const &func) {
    TICK(fill);
    // for (size_t i = 0; i < arr.size(); i++) {
    //     arr[i] = func(i);
    // }
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0,arr.size()),
        [&](tbb::blocked_range<size_t> r){
            for ( size_t i=r.begin();i<r.end();i++){
                arr[i]=func(i);
            }
        }
    );
    TOCK(fill);
    return arr;
}

template <class T>
void saxpy(T a, std::vector<T> &x, std::vector<T> const &y) {
    TICK(saxpy);
    // for (size_t i = 0; i < x.size(); i++) {
    //    x[i] = a * x[i] + y[i];
    // }
    tbb::parallel_for(
        tbb::blocked_range<size_t> (0 , x.size()),
        [&](tbb::blocked_range<size_t> r){
            for ( size_t i=r.begin();i<r.end();i++){
                x[i] = a * x[i] + y[i];
            }
        }
    );

    TOCK(saxpy);
}

template <class T>
T sqrtdot(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(sqrtdot);
    // T ret = 0;
    // for (size_t i = 0; i < std::min(x.size(), y.size()); i++) {
    //     ret += x[i] * y[i];
    // }
    T ret=tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0,std::min(x.size(), y.size())),T(0),
        [&](tbb::blocked_range<size_t> r,T local_ret){
            for (size_t i=r.begin();i<r.end();i++){
                local_ret+=x[i]*y[i];
            }
            return local_ret;
        },
        [](T a,T b){
            return a+b;
        }
    );
    ret = std::sqrt(ret);
    TOCK(sqrtdot);
    return ret;
}

template <class T>
T minvalue(std::vector<T> const &x) {
    TICK(minvalue);
    // T ret = x[0];
    // for (size_t i = 1; i < x.size(); i++) {
    //     if (x[i] < ret)
    //         ret = x[i];
    // }
    T ret=tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0,x.size()),x[0],
        [&](tbb::blocked_range<size_t> r,T local_ret){
            for(size_t i=r.begin();i<r.end();i++){
                if(x[i]<local_ret)
                    local_ret=x[i];
            }
            return local_ret;
        },
        [](T a,T b){
            return a<b?a:b;
        }
    );
    TOCK(minvalue);
    return ret;
}

template <class T>
std::vector<pod<T>> magicfilter(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(magicfilter);
    // std::vector<T> res;
    // for (size_t i = 0; i < std::min(x.size(), y.size()); i++) {
    //     if (x[i] > y[i]) {
    //         res.push_back(x[i]);
    //     } else if (y[i] > x[i] && y[i] > 0.5f) {
    //         res.push_back(y[i]);
    //         res.push_back(x[i] * y[i]);
    //     }
    // }
    std::vector<pod<T>> res(2*std::min(x.size(), y.size()));
    std::atomic<size_t> index{0};
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0,std::min(x.size(), y.size())),
        [&](tbb::blocked_range<size_t> r){
            std::vector<pod<T>> local_res;
            // local_res.reserve(y.size()/NUM_THREADS);
            for(size_t i=r.begin();i<r.end();i++){
                if (x[i] > y[i]) {
                    local_res.push_back(x[i]);
                } else if (y[i] > x[i] && y[i] > 0.5f) {
                    local_res.push_back(y[i]);
                    local_res.push_back(x[i] * y[i]);
                }
            }
            int beg=index.fetch_add(local_res.size());
            std::memcpy(&res[beg],&local_res[0],local_res.size());
        }
    );
    res.resize(index);
    TOCK(magicfilter);
    return res;
}

template <class T>
T scanner(std::vector<T> &x) {
    TICK(scanner);
    // T ret = 0;
    // for (size_t i = 0; i < x.size(); i++) {
    //     ret += x[i];
    //     x[i] = ret;
    // }

    //实测下面的手动划分task方式比auto_partiioner的parallel_scan更快
    // float ret = tbb::parallel_scan(tbb::blocked_range<size_t>(0, x.size()), (float)0,
    // [&] (tbb::blocked_range<size_t> r, float local_res, auto is_final) {
    //     for (size_t i = r.begin(); i < r.end(); i++) {
    //         local_res += x[i];
    //         if (is_final) {
    //             x[i] = local_res;
    //         }
    //     }
    //     return local_res;
    // }, [] (float x, float y) {
    //     return x + y;
    // });

    //手动划分任务区间
    tbb::task_group tg;
    std::vector<T> local_res(NUM_THREADS);
    for (size_t k=0;k<NUM_THREADS;k++){
        size_t beg=k*(x.size()+NUM_THREADS-1)/NUM_THREADS;//应该向上取整
        size_t end=std::min((k+1)*(x.size()+NUM_THREADS-1)/NUM_THREADS,x.size());
        tg.run(
            [&,k,beg,end](){
                T tmp=0.f;
                for(size_t i=beg;i<end;i++){
                    tmp+=x[i];
                    x[i]=tmp;
                }
                local_res[k]=tmp;
            }
        );
    }
    tg.wait();
    T pre_sum=0.f;
    for (size_t k=0;k<NUM_THREADS;k++){
        pre_sum+=local_res[k];
        local_res[k]=pre_sum;
    }
    for(size_t k=1;k<NUM_THREADS;k++){
        size_t beg=k*(x.size()+NUM_THREADS-1)/NUM_THREADS;
        size_t end=std::min((k+1)*(x.size()+NUM_THREADS-1)/NUM_THREADS,x.size());
        tg.run(
            [&,k,beg,end](){
                for(size_t i=beg;i<end;i++){
                    x[i]+=local_res[k-1];
                }
            }
        );
    }
    tg.wait();


    TOCK(scanner);
    return local_res[NUM_THREADS-1];
    // return ret;
}

int main() {
    size_t n = 1<<26;
    std::vector<float> x(n);
    std::vector<float> y(n);

    tbb::task_scheduler_init init(NUM_THREADS);
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
