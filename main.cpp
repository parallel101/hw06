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
#include <tbb/parallel_scan.h>
#include <tbb/concurrent_vector.h>


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
auto magicfilter(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(magicfilter);
    tbb::concurrent_vector<T> tmp_res;

    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, std::min(x.size(), y.size())), [&](auto r) {
        std::vector<T> local_res;
        local_res.reserve(r.size());

        for (auto i = r.begin(); i != r.end(); ++i) {
            if (x[i] > y[i]) {
                local_res.push_back(x[i]);
            } else if (y[i] > x[i] && y[i] > 0.5f) {
                local_res.push_back(y[i]);
                local_res.push_back(x[i] * y[i]);
            }
        }

        auto it = tmp_res.grow_by(local_res.size());
        std::copy(local_res.begin(), local_res.end(), it);
    });

    std::vector<pod<T>> res;
    res.reserve(tmp_res.size());
    std::copy(tmp_res.begin(), tmp_res.end(), std::back_inserter(res));

    TOCK(magicfilter);
    return res;
}

template<class T>
T scanner(std::vector<T> &x) {
    TICK(scanner);

    auto ret = tbb::parallel_scan(tbb::blocked_range<std::size_t>(0, x.size()), T{},
                                  [&](auto r, auto local_res, auto is_final) {
                                      for (auto i = r.begin(); i != r.end(); ++i) {
                                          local_res += x[i];
                                          if (is_final) {
                                              x[i] = local_res;
                                          }
                                      }
                                      return local_res;
                                  },
                                  [](auto x, auto y) { return x + y; });

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
