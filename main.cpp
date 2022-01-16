#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <numeric>
#include <mutex>
#include <algorithm>
#include <tbb/parallel_scan.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_sort.h>
#include <tbb/blocked_range.h>
#include <tbb/spin_mutex.h>
#include "ticktock.h"


// TODO: 并行化所有这些 for 循环
// 测试平台：Windows 10 专业版 20H2
// CPU:     Intel(R) Core(TM) i5-5200U CPU @ 2.20GHz   2.19 GHz
// 编译工具：VS2017 C++17
// TBB版本： 2020_U3#6

// PARALLEL_TYPE_TEST 
#define	PARALLEL_TYPE_NULL 0
#define	PARALLEL_TYPE_FOR  1
#define	PARALLEL_TYPE_RANDOM_ALLOC 2
// 用于切换并行方案
#define SPTT PARALLEL_TYPE_RANDOM_ALLOC

// 排序测试
// PARALLEL_SORT_TEST
#define PARALLEL_SORT_NULL 0
#define PARALLEL_FOR_SORT  1
#define PARALLEL_SORT_SORT 2
// 用于切换并行方案
#define SPST  PARALLEL_FOR_SORT

template <class T, class Func>
std::vector<T> fill(std::vector<T> &arr, Func const &func) {
    TICK(fill);

#if (SPTT == PARALLEL_TYPE_FOR)
    // 并行赋值 1.7s左右
    std::cout << "PARALLEL_TYPE_FOR" << std::endl;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, arr.size()),
    [&](tbb::blocked_range<size_t> r) {
        for (size_t i = r.begin(); i < r.end(); ++i)
        {
            arr[i] = func(i);
        }
    });
#elif (SPTT == PARALLEL_TYPE_RANDOM_ALLOC)
    // 建立四个线程 任务域
    // 使用tbb::affinity_partitioner自动负载均衡，第二次比第一次快0.4s。
    // 使用tbb::simple_partitioner 8s左右
    // 使用tbb::auto_partitioner   1.9s左右
    tbb::task_arena ta(4);
    ta.execute([&] {
        tbb::affinity_partitioner affinity;
        tbb::parallel_for(tbb::blocked_range<size_t>(0, arr.size()),
            [&](tbb::blocked_range<size_t> r) 
        {
            for (size_t i = r.begin(); i < r.end(); ++i)
            {
                arr[i] = func(i);
            }
        }, affinity);
    });
#else
    for (size_t i = 0; i < arr.size(); i++) {
        arr[i] = func(i);
    }
#endif

    TOCK(fill);
    return arr;
}

template <class T>
void saxpy(T a, std::vector<T> &x, std::vector<T> const &y) {
    TICK(saxpy);
    // 并行 0.081 -> 0.049提升0.032s
    auto mincnt = std::min<T>(x.size(), y.size());
    tbb::parallel_for(tbb::blocked_range<size_t>(0, mincnt),
        [&](tbb::blocked_range<size_t> r) {
        for (size_t i = r.begin(); i < r.end(); ++i)
        {
            // 直接赋值开销小
            x[i] = a * x[i] + y[i];
        }
    });
    TOCK(saxpy);
}

template <class T>
T sqrtdot(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(sqrtdot);
    T ret = 0;
    // 避免循环中重复计算vector的size, 提升0.05s。
    auto mincnt = std::min<T>(x.size(), y.size());
    tbb::spin_mutex spin_mtx;
    
    // 求和需要考虑线程同步
#if (SPTT == PARALLEL_TYPE_FOR)
    // 并行 0.16-> 0.07s左右。
    tbb::parallel_for(tbb::blocked_range<size_t>(0, mincnt),
        [&](tbb::blocked_range<size_t> r)
    {
        // 以下语句内存不足, 不采用小彭老师的推荐方案
        // std::vector<T> temp_a(r.size()); 
        T val;
        T total{ 0 };
        for (size_t i = r.begin(); i < r.end(); ++i)
        {
            val = x[i] * y[i];
            if (val > 0)
            {
                total += val;;
            }
        }

        std::lock_guard lck(spin_mtx);
        ret += total;
    }
    );
#elif (SPTT == PARALLEL_TYPE_RANDOM_ALLOC)
    // 建立四个线程 
    // 使用tbb::affinity_partitioner自动负载均衡，0.16-> 0.05s左右。
    // 使用tbb::simple_partitioner 0.16-> 0.052s左右
    // 使用tbb::auto_partitioner   0.16-> 0.05s左右
    tbb::task_arena ta(4);
    ta.execute([&] {
        tbb::affinity_partitioner affinity;
        tbb::parallel_for(tbb::blocked_range<size_t>(0, mincnt),
            [&](tbb::blocked_range<size_t> r)
        {
            T val;
            T total{ 0 };
            for (size_t i = r.begin(); i < r.end(); ++i)
            {
                val = x[i] * y[i];
                if (val > 0)
                {
                    total += val;;
                }
            }

            std::lock_guard lck(spin_mtx);
            ret += total;
        }, affinity);
    });

#else
    for (size_t i = 0; i < mincnt; i++) {
        ret += x[i] * y[i];
    }
#endif

    ret = std::sqrt(ret);
    TOCK(sqrtdot);
    return ret;
}

template <class T>
T minvalue(std::vector<T> const &x) {
    TICK(minvalue);
    
    T ret = x[0];
#if (SPST == PARALLEL_FOR_SORT)
    // 采用parallel for  0.092 -> 0.033s 左右
    tbb::spin_mutex spin_mtx;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, x.size()),
        [&](tbb::blocked_range<size_t> r) {
        T min_val{x[r.begin()]};
        for (size_t i = r.begin(); i < r.end(); ++i)
        {
            if (x[i] < min_val)
            {
                min_val = x[i];
            }
        }
        std::lock_guard lck(spin_mtx);
        if (min_val < ret)
        {
            ret = min_val;
        }
    });
    
#elif (SPST == PARALLEL_SORT_SORT)
    // 采用parallel sort  0.092 -> 4.7s 左右
    std::vector vec_temp = std::move(x);
    tbb::parallel_sort(vec_temp.begin(), vec_temp.end(), std::less<T>{});
    ret = vec_temp[0];
    
#else
    // 非并行取最小值
    for (size_t i = 1; i < x.size(); i++) {
        if (x[i] < ret)
            ret = x[i];
    }
#endif

    TOCK(minvalue);
    return ret;
}

template <class T>
std::vector<T> magicfilter(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(magicfilter);
    std::mutex mtx;
    auto mincnt = std::min<T>(x.size(), y.size());

    // 无法事先预计返回数据长度
    // 预分配空间反而会导致性能降低
    //std::vector<T> res(mincnt);
    std::vector<T> res;

#if 1
    // 优化前后对比 0.8s -> 0.56s
    tbb::task_arena ta(4);
    ta.execute([&] {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, mincnt),
            [&](tbb::blocked_range<size_t> r)
        {
            std::vector<T> temp;
            for (size_t i = r.begin(); i < r.end(); ++i)
            {
                if (x[i] > y[i]) {
                    temp.push_back(x[i]);
                }
                else if (y[i] > 0.5f && y[i] < x[i]) {
                    // 主观预计(y[i] > 0.5f) 为false的概率大于(y[i] < x[i])
                    // 故将其放在前面判断
                    temp.push_back(y[i]);
                    temp.push_back(x[i] * y[i]);
                }
            }
            std::lock_guard lck(mtx);
            std::copy(temp.begin(), temp.end(), std::back_inserter(res));
        }, tbb::auto_partitioner{});
    });

#else
    
    for (size_t i = 0; i < mincnt; i++) {
        if (x[i] > y[i]) {
            res.push_back(x[i]);
        }
        else if (y[i] > x[i] && y[i] > 0.5f) {
            res.push_back(y[i]);
            res.push_back(x[i] * y[i]);
        }
    }

#endif

    TOCK(magicfilter);
    return res;
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

    return 0;
}
