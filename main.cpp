#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "ticktock.h"
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include "pod.h"
#include <atomic>
#include <tbb/parallel_scan.h>
// TODO: 并行化所有这些 for 循环
// 使用任务域指定使用的线程数4

template <class T, class Func>
std::vector<T> fill(std::vector<T> &arr, Func const &func) {
    TICK(fill);
    // for (size_t i = 0; i < arr.size(); i++) {
    //     arr[i] = func(i);
    // }

    // fiil(x) 0.951s -> 0.258s 提高3.69倍
    // fill(y) 0.978s -> 0.253s 提高3.86倍
    tbb::task_arena ta(4);
    ta.execute([&] {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, arr.size()),
        [&] (tbb::blocked_range<size_t> r) {
            for (size_t i = r.begin(); i < r.end(); i++) {
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
    // for (size_t i = 0; i < x.size(); i++) {
    //     x[i] = a * x[i] + y[i];
    // }
    // 0.042s -> 0.021s  提高2倍
    tbb::task_arena ta(4);
    ta.execute([&] {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, x.size()),
        [&] (tbb::blocked_range<size_t> r) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                x[i] = a * x[i] + y[i];
            }
        }, tbb::auto_partitioner{});
    });
    TOCK(saxpy);
}

template <class T>
T sqrtdot(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(sqrtdot);
    // 串行reduce float类型会出现浮点误差
    // T ret = 0;
    // for (size_t i = 0; i < std::min(x.size(), y.size()); i++) {
    //     ret += x[i] * y[i];
    // }

    // 0.0916s -> 0.0214s 提高3.88倍
    T ret = 0;
    tbb::task_arena ta(4);

    ta.execute([&] {
        ret = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, std::min(x.size(), y.size())), T(0),
        [&] (tbb::blocked_range<size_t> r, T local_ret) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                local_ret += x[i] * y[i];
            }
            return local_ret;
        }, [] (T x, T y) {
            return x + y;
        });
    });
    ret = std::sqrt(ret);
    TOCK(sqrtdot);
    return ret;
}

// 求最小值也是reduce
template <class T>
T minvalue(std::vector<T> const &x) {
    TICK(minvalue);
    // T ret = x[0];
    // for (size_t i = 1; i < x.size(); i++) {
    //     if (x[i] < ret)
    //         ret = x[i];
    // }

    // 0.092s -> 0.023s 提高4倍
    T ret = x[0];
    tbb::task_arena ta(4);

    ta.execute([&] {
        ret = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, x.size()), x[1],
        [&] (tbb::blocked_range<size_t> r, T local_ret) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                if (x[i] < local_ret)
                    local_ret = x[i];
            }
            return local_ret;
        }, [] (T x, T y) {
            return std::min(x, y);
        });
    });
    TOCK(minvalue);
    return ret;
}

// 这里使用了小彭老师写的pod.h 使用resize不初始化其中的值
template <class T>
auto magicfilter(std::vector<T> const &x, std::vector<T> const &y) {
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

    // 0.41s -> 0.05s 提高8倍
    std::vector<pod<T>> res;  // 避免初始化
    std::atomic<size_t> res_size = 0;
    size_t n = std::min(x.size(), y.size());
    res.resize(n);

    tbb::task_arena ta(4);

    ta.execute([&] {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
        [&] (tbb::blocked_range<size_t> r) {
            std::vector<pod<T>> local_res(r.size());
            size_t lrsize = 0;
            for (size_t i = r.begin(); i < r.end(); i++) {
                if (x[i] > y[i]) {
                    local_res[lrsize++] = x[i];
                } else if (y[i] > x[i] && y[i] > 0.5f) {
                    local_res[lrsize++] = y[i];
                    local_res[lrsize++] = x[i] + y[i];
                }
            }
            size_t base = res_size.fetch_add(lrsize);
            for (size_t i = 0; i < lrsize; i++) {
                res[base + i] = local_res[i];
            }
        });
    });
    res.resize(res_size);
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

    // 0.09s -> 0.047s 提高1.91倍
    T ret = 0;
    tbb::task_arena ta(4);

    ta.execute([&] {
        ret = tbb::parallel_scan(tbb::blocked_range<size_t>(0, x.size()), T(0),
        [&] (tbb::blocked_range<size_t> r, T local_ret, auto is_final) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                local_ret += x[i];
                if (is_final) {
                    x[i] = local_ret;
                }
            }
            return local_ret;
        }, [] (T x, T y) {
            return x + y;
        });
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
