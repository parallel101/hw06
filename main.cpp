#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "pod.h"
#include "ticktock.h"
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>


// TODO: 并行化所有这些 for 循环

template<class T, class Func>
std::vector<T> fill(std::vector<T> &arr, Func const &func) {
    TICK(fill);
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, arr.size()),
                      [&](auto r) {
                          for (auto i = r.begin(); i != r.end(); ++i) {
                              arr[i] = func(i);
                          }
                      });
    TOCK(fill);
    return arr;
}

template<class T>
void saxpy(T a, std::vector<pod<T>> &x, std::vector<pod<T>> const &y) {
    TICK(saxpy);

    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, x.size()),
                      [&](auto r) {
                          for (auto i = r.begin(); i != r.end(); ++i) {
                              x[i] = a * x[i] + y[i];
                          }
                      });

    TOCK(saxpy);
}

template<class T>
T sqrtdot(std::vector<pod<T>> const &x, std::vector<pod<T>> const &y) {
    TICK(sqrtdot);
    auto ret = tbb::parallel_reduce(tbb::blocked_range<std::size_t>(0, std::min(x.size(), y.size())), 0.f,
                                    [&](auto r, auto local_res) {
                                        for (auto i = r.begin(); i != r.end(); ++i) {
                                            local_res += x[i] * y[i];
                                        }
                                        return local_res;
                                    },
                                    [](auto x, auto y) {
                                        return x + y;
                                    });

    ret = std::sqrt(ret);
    TOCK(sqrtdot);
    return ret;
}

template<class T>
T minvalue(std::vector<pod<T>> const &x) {
    TICK(minvalue);

    auto ret = tbb::parallel_reduce(tbb::blocked_range<std::size_t>(0, x.size()), x[0],
                                    [&](auto r, auto local_min_value) {

                                        for (auto i = r.begin(); i != r.end(); ++i) {
                                            local_min_value = std::min(local_min_value, x[i]);
                                        }

                                        return local_min_value;
                                    },
                                    [](auto x, auto y) {
                                        return std::min(x, y);
                                    });

    TOCK(minvalue);
    return ret;
}

template<class T>
std::vector<T> magicfilter(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(magicfilter);
    std::vector<T> res;
    for (size_t i = 0; i < std::min(x.size(), y.size()); i++) {
        if (x[i] > y[i]) {
            res.push_back(x[i]);
        } else if (y[i] > x[i] && y[i] > 0.5f) {
            res.push_back(y[i]);
            res.push_back(x[i] * y[i]);
        }
    }
    TOCK(magicfilter);
    return res;
}

template<class T>
T scanner(std::vector<T> &x) {
    TICK(scanner);
    T ret = 0;
    for (size_t i = 0; i < x.size(); i++) {
        ret += x[i];
        x[i] = ret;
    }
    TOCK(scanner);
    return ret;
}

int main() {
    size_t n = 1 << 26;
    std::vector<pod<float>> x(n);
    std::vector<pod<float>> y(n);

    fill(x, [&](size_t i) { return std::sin(i); });
    fill(y, [&](size_t i) { return std::cos(i); });

    saxpy(0.5f, x, y);

    std::cout << sqrtdot(x, y) << std::endl;
    std::cout << minvalue(x) << std::endl;

    auto arr = magicfilter(x, y);
    std::cout << arr.size() << std::endl;

    scanner(x);
    std::cout << std::reduce(x.begin(), x.end()) << std::endl;

    return 0;
}
