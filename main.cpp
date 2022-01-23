#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <atomic>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>


#include "ticktock.h"
#include "pod.h"
// TODO: 并行化所有这些 for 循环

template <class T, class Func>
std::vector<T> fill(std::vector<T> &arr, Func const &func) {
    TICK(fill);
    //直接并行 ，平均时间：0.0018s
    tbb::parallel_for(tbb::blocked_range<size_t>(0, arr.size()), [&](const tbb::blocked_range<size_t> &r) {
        for (size_t i = r.begin(); i != r.end(); i++) {
            arr[i] = func(i);
        }
    });
    TOCK(fill);
    return arr;
}

template <class T>
void saxpy(T a, std::vector<T> &x, std::vector<T> const &y) {
    TICK(saxpy);
    //直接并行 ，平均时间：0.05s
    tbb::parallel_for(tbb::blocked_range<size_t>(0, x.size()), [&](const tbb::blocked_range<size_t> &r) {
        for (size_t i = r.begin(); i != r.end(); i++) {
            x[i] = a * x[i] + y[i];
        }
    });
    TOCK(saxpy);
}

template <class T>
T sqrtdot(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(sqrtdot);
    //并行缩并 ，平均时间：0.032s
    T ret = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, x.size()), (T)0,[&](tbb::blocked_range<size_t> &r, T local_res) {
        for (size_t i = r.begin(); i != r.end(); i++) {
            local_res += x[i] * y[i];
        }
        return local_res;
    }, [](T a, T b) {
        return a + b;
    });
    ret = std::sqrt(ret);
    TOCK(sqrtdot);
    return ret;
}

template <class T>
T minvalue(std::vector<T> const &x) {
    TICK(minvalue);
    //并行缩并求最小值 ，平均时间：0.015s
    T ret = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, x.size()), (T)0,[&](tbb::blocked_range<size_t> &r, T local_res) {
        for (size_t i = r.begin(); i != r.end(); i++) {
            if (x[i] < local_res)
                local_res = x[i];
        }
        return local_res;
    }, [](T a, T b) {
        return std::min(a, b);
    });
    TOCK(minvalue);
    return ret;
}

template <class T>
auto magicfilter(std::vector<T> const &x, std::vector<T> const &y) {
    std::vector<pod<T>> res;
    std::atomic<size_t> res_size = 0;
    //使用彭老师的头文件，平均时间：0.06s
    TICK(magicfilter);
    res.resize(x.size());
    tbb::parallel_for(tbb::blocked_range<size_t>(0, x.size()),
            [&](const tbb::blocked_range<size_t> &r) {
        std::vector<pod<T>> local_a(r.size());
        size_t lasize = 0;
        for (size_t i = r.begin(); i != r.end(); i++) {
            if(x[i]>y[i]){
                local_a[lasize++] = x[i];
            } else if(y[i] > x[i] && y[i] >0.5f){
                local_a[lasize++] = y[i];
                local_a[lasize++] = x[i] * y[i];
            }
        }
        size_t base = res_size.fetch_add(lasize);
        for(size_t i=0;i<lasize;i++){
            res[base+i] = local_a[i];
        }
    });
    res.resize(res_size);
    TOCK(magicfilter);
    return res;
}

template <class T>
T scanner(std::vector<T> &x) {
    TICK(scanner);
    //平均时间：0.06s
    T ret = 0;
    tbb::task_arena ta(4);
    ta.execute([&] {
        ret = tbb::parallel_scan(tbb::blocked_range<size_t>(0, x.size()),T(0), [&](const tbb::blocked_range<size_t> &r, T local_res,auto is_final) {
            for (size_t i = r.begin(); i != r.end(); i++) {
                local_res += x[i];
                if(is_final)x[i] = local_res;
            }
            return local_res;
        },[] (T x, T y) {
            return x + y;
        },tbb::auto_partitioner());
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
