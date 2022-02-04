#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
#include <thread>
#include <vector>

#include "pod.h"
#include "ticktock.h"

// TODO: 并行化所有这些 for 循环

template <class T, class Func>
std::vector<T> fill(std::vector<T>& arr, Func const& func) {
  TICK(fill);
  const int n = arr.size();
  tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
                    [&](tbb::blocked_range<size_t> r) {
                      for (size_t i = r.begin(); i < r.end(); ++i) {
                        arr[i] = func(i);
                      }
                    });
  TOCK(fill);
  return arr;
}

template <class t>
void saxpy(t a, std::vector<t>& x, std::vector<t> const& y) {
  TICK(saxpy);
  const int n = x.size();
  tbb::task_arena ta(std::thread::hardware_concurrency());
  tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
                    [&](tbb::blocked_range<size_t> r) {
                      for (size_t i = r.begin(); i < r.end(); ++i) {
                        x[i] = a * x[i] + y[i];
                      }
                    });
  TOCK(saxpy);
}

template <class T>
T sqrtdot(std::vector<T> const& x, std::vector<T> const& y) {
  TICK(sqrtdot);
  const int n = std::min(x.size(), y.size());
  // tbb::parallel_deterministic_reduce(tbb::blocked_range<size_t>(0, n),
  T ret = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, n), static_cast<T>(0),
      [&](tbb::blocked_range<size_t> r, T local_res) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
          local_res += x[i] * y[i];
        }
        return local_res;
      },
      [](T x, T y) { return x + y; });
  ret = std::sqrt(ret);
  TOCK(sqrtdot);
  return ret;
}

template <class T>
T minvalue(std::vector<T> const& x) {
  TICK(minvalue);
  const size_t n = x.size();
  T ret = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, n), x[0],
      [&](tbb::blocked_range<size_t> r, T local_res) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
          local_res = std::min(local_res, x[i]);
        }
        return local_res;
      },
      [](T a, T b) { return std::min(a, b); });
  TOCK(minvalue);
  return ret;
}

template <class T>
std::vector<pod<T>> magicfilter(std::vector<T> const& x,
                                std::vector<T> const& y) {
  TICK(magicfilter);
  std::vector<pod<T>> res;
  const int n = std::min(x.size(), y.size());
  res.resize(n * 2);
  std::atomic<size_t> idx = 0;
  tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
                    [&](tbb::blocked_range<size_t> r) {
                      std::vector<pod<T>> local_res(r.size() * 2);
                      size_t local_idx = 0;
                      for (size_t i = r.begin(); i < r.end(); ++i) {
                        if (x[i] > y[i]) {
                          local_res[local_idx++] = x[i];
                        } else if (y[i] > x[i] && y[i] > 0.5f) {
                          local_res[local_idx++] = y[i];
                          local_res[local_idx++] = x[i] * y[i];
                        }
                      }
                      size_t base = idx.fetch_add(local_idx);
                      for (size_t i = 0; i < local_idx; ++i) {
                        res[base + i] = local_res[i];
                      }
                    });
  res.resize(idx);
  TOCK(magicfilter);
  return res;
}

template <class T>
T scanner(std::vector<T>& x) {
  TICK(scanner);
  const int n = x.size();
  T ret = tbb::parallel_scan(
      tbb::blocked_range<size_t>(0, n), static_cast<T>(0),
      [&](tbb::blocked_range<size_t> r, T local_res, auto is_final) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
          local_res += x[i];
          if (is_final) {
            x[i] = local_res;
          }
        }
        return local_res;
      },
      [](T x, T y) { return x + y; });
  TOCK(scanner);
  return ret;
}

int main() {
  size_t n = 1 << 26;
  std::vector<float> x(n);
  std::vector<float> y(n);

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
