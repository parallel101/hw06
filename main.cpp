#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <numeric>
#include "ticktock.h"
#include <tbb/tbb.h>
#include <atomic>

// TODO: 并行化所有这些 for 循环

template <class T, class Func>
std::vector<T> fill(std::vector<T> &arr, Func const &func) {
    TICK(fill);
    tbb::parallel_for(0,(int)arr.size(),[&](auto i){
        arr[i] = func(i);
    });
    TOCK(fill);
    return arr;
}

template <class T>
void saxpy(T a, std::vector<T> &x, std::vector<T> const &y) {
    TICK(saxpy);
    tbb::parallel_for(0,(int)x.size(),[&](auto i){
        x[i] = a * x[i] + y[i];
    });
    TOCK(saxpy);
}

template <class T>
T sqrtdot(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(sqrtdot);
    T ret = tbb::parallel_reduce(tbb::blocked_range(0,(int)std::min(x.size(), y.size())),T{},[&](auto r,auto f){
        for (int i = r.begin();i < r.end();i++)
            f += x[i] * y[i];
        return f;
    },[](auto a,auto b){return a+b;});
    ret = std::sqrt(ret);
    TOCK(sqrtdot);
    return ret;
}

template <class T>
T minvalue(std::vector<T> const &x) {
    TICK(minvalue);
    T ret = tbb::parallel_reduce(tbb::blocked_range(1,(int)x.size()),x[0],[&](auto r,auto f){
        for (int i = r.begin();i < r.end();i++)
            f = std::min(f,x[i]);
        return f;
    },[](auto a,auto b){return std::min(a,b);});
    TOCK(minvalue);
    return ret;
}

template <class T>
auto magicfilter(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(magicfilter);
    // std::vector<T> res(3 * std::min(x.size(), y.size()));
    tbb::concurrent_vector<T> res;
    // std::atomic<int> s = 0;
    tbb::parallel_for(tbb::blocked_range(0,(int)std::min(x.size(), y.size())),[&](auto r){
        std::vector<T> tmp;
        for (size_t i = r.begin(); i < r.end(); i++) {
            if (x[i] > y[i]) {
                tmp.push_back(x[i]);
            } else if (y[i] > x[i] && y[i] > 0.5f) {
                tmp.push_back(y[i]);
                tmp.push_back(x[i] * y[i]);
            }
        }
        auto it = res.grow_by(tmp.size());
        std::copy(tmp.begin(),tmp.end(),it);
    });
    TOCK(magicfilter);
    return res;
}

template <class T>
T scanner(std::vector<T> &x) {
    TICK(scanner);
    T ret = tbb::parallel_scan(tbb::blocked_range(0,(int)x.size()),T{},[&](auto r,auto f,auto final){
        for (int i = r.begin();i < r.end();i++){
            f += x[i];
            if (final)
                x[i] = f;
        }
        return f;
    },[](auto a,auto b){return a + b;});
    TOCK(scanner);
    return ret;
}

int main() {
    size_t n = 1<<26;
    std::vector<double> x(n);
    std::vector<double> y(n);
    

    fill(x, [&] (size_t i) { return std::sin(i); });
    fill(y, [&] (size_t i) { return std::cos(i); });

    saxpy(0.5, x, y);

    std::cout << sqrtdot(x, y) << std::endl;
    std::cout << minvalue(x) << std::endl;

    auto arr = magicfilter(x, y);
    std::cout << arr.size() << std::endl;

    scanner(x);
    std::cout << std::reduce(x.begin(), x.end()) << std::endl;
    
    return 0;
}
