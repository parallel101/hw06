#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "ticktock.h"

#include <tbb/tbb.h>
#include "pod.h"

// TODO: 并行化所有这些 for 循环

template <class T, class Func>
std::vector<T> fill(std::vector<T> &arr, Func const &func)
{
    TICK(fill);
    size_t n = arr.size();
    tbb::task_arena ta(4);
    ta.execute(
        [&] {
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, n), 
                [&](const tbb::blocked_range<size_t> &r)
                {
                    for(size_t i = r.begin(); i < r.end(); i++)
                        arr[i] = func(i);
                }
            );
        }
    );
    TOCK(fill);
    return arr;
}

template <class T>
void saxpy(T a, std::vector<T> &x, std::vector<T> const &y)
{
    TICK(saxpy);
    size_t n = x.size();
    tbb::task_arena ta(4);
    ta.execute(
        [&] {
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, n),
                [&](const tbb::blocked_range<size_t> &r)
                {
                    for(size_t i = r.begin(); i < r.end(); i++)
                        x[i] = a * x[i] + y[i];
                }
            );
        }
    );
    TOCK(saxpy);
}

template <class T>
T sqrtdot(std::vector<T> const &x, std::vector<T> const &y)
{
    TICK(sqrtdot);
    size_t n = x.size();
    tbb::task_arena ta(4);
    T ret;
    ta.execute(
        [&]{
            ret = tbb::parallel_reduce(
                tbb::blocked_range<size_t>(0, n), (T)0,
                [&](const tbb::blocked_range<size_t> &r, T ans)
                {
                    for(size_t i = r.begin(); i < r.end(); i++)
                        ans += x[i] * y[i];
                    return ans;
                },
                [](T a, T b)
                {
                    return a + b;
                }
            );
        }
    );
    ret = std::sqrt(ret);
    TOCK(sqrtdot);
    return ret;
}

template <class T>
T minvalue(std::vector<T> const &x)
{
    TICK(minvalue);
    size_t n = x.size();
    tbb::task_arena ta(4);
    T ret;
    ta.execute(
        [&] {
            ret = tbb::parallel_reduce(
                tbb::blocked_range<size_t>(0, n), (T)x[0],
                [&](const tbb::blocked_range<size_t> &r, T ans)
                {
                    for(size_t i = r.begin(); i < r.end(); i++)
                        ans = std::min(ans, x[i]);
                    return ans;
                },
                [](T a, T b)
                {
                    return std::min(a, b);
                }
            );
        }
    );
    TOCK(minvalue);
    return ret;
}

template <class T>
auto magicfilter(std::vector<T> const &x, std::vector<T> const &y)
{
    TICK(magicfilter);
    tbb::concurrent_vector<pod<T> > res;
    size_t n = x.size();
    tbb::task_arena ta(4);
    ta.execute(
        [&] {
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, n),
                [&](const tbb::blocked_range<size_t> &r)
                {
                    std::vector<pod<T> > ans;
                    ans.reserve(r.size());
                    for(size_t i = r.begin(); i < r.end(); i++)
                    {
                        if(x[i] > y[i])
                            ans.push_back(x[i]);
                        else if(y[i] > x[i] && y[i] > 0.5f)
                        {
                            ans.push_back(y[i]);
                            ans.push_back(x[i] * y[i]);
                        }
                    }
                    auto it = res.grow_by(ans.size());
                    std::copy(ans.begin(), ans.end(), it);
                },
                tbb::static_partitioner{}
            );
        }
    );
    TOCK(magicfilter);
    return res;
}

template <class T>
T scanner(std::vector<T> &x)
{
    TICK(scanner);
    size_t n = x.size();
    tbb::task_arena ta(4);
    T ret;
    ta.execute(
        [&] {
            ret = tbb::parallel_scan(
                tbb::blocked_range<size_t>(0, n), (T)0,
                [&](const tbb::blocked_range<size_t> &r, float ans, auto is_final)
                {
                    for(size_t i = r.begin(); i < r.end(); i++)
                    {
                        ans += x[i];
                        if(is_final)
                            x[i] = ans;
                    }
                    return ans;
                },
                [](T a, T b)
                {
                    return a + b;
                }
            );
        }
    );
    TOCK(scanner);
    return ret;
}

int main()
{
    size_t n = 1 << 26;
    std::vector<float> x(n);
    std::vector<float> y(n);

    fill(x, [&](size_t i)
         { return std::sin(i); });
    fill(y, [&](size_t i)
         { return std::cos(i); });

    saxpy(0.5f, x, y);

    std::cout << sqrtdot(x, y) << std::endl;
    std::cout << minvalue(x) << std::endl;

    auto arr = magicfilter(x, y);
    std::cout << arr.size() << std::endl;

    scanner(x);
    std::cout << std::reduce(x.begin(), x.end()) << std::endl;

    return 0;
}
