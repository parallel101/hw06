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
#include <tbb/parallel_for_each.h>
#include <tbb/concurrent_vector.h>
#include <tbb/task_arena.h>
#include "ticktock.h"
#include "pod.h"

// TODO: 并行化所有这些 for 循环

template <class T, class Func>
std::vector<T> fill(std::vector<T> &arr, Func const &func) {
    TICK(fill);
	//tbb::task_arena ta(4);
	//ta.execute([&] {
		tbb::parallel_for(tbb::blocked_range<size_t>(0, arr.size()),
		[&](tbb::blocked_range<size_t> r){
			for(size_t i=std::begin(r);i<std::end(r);i++)
            arr[i]=func(i);
		});
	//});
    TOCK(fill);
    return arr;
}
// template <class T, class Func>
// std::vector<T> fill(std::vector<T> &arr, Func const &func) {
//     TICK(fill);
//     for(size_t i=0;i<arr.size();i++)
//     {
//         arr[i]=func(i);
//     }
//     TOCK(fill);
//     return arr;
// }

template <class T>
void saxpy(T a, std::vector<T> &x, std::vector<T> const &y) {
    TICK(saxpy);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, x.size()),
		[&](tbb::blocked_range<size_t> r) {
			for (size_t i = std::begin(r); i < std::end(r); i++)
				x[i] = a*x[i]+y[i];
		});
    TOCK(saxpy);
}

template <class T>
T sqrtdot(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(sqrtdot);
	T ret = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, std::min(x.size(), y.size())), (T)0, [&](tbb::blocked_range<size_t> r, T local_sum) {
		for (size_t i = std::begin(r); i != std::end(r); i++)
		{
			local_sum += x[i]*y[i];
		}
		return local_sum;
		},
		[](T x, T y) {
			return x+y;
		});
    ret = std::sqrt(ret);
    TOCK(sqrtdot);
    return ret;
}

template <class T>
T minvalue(std::vector<T> const &x) {
    TICK(minvalue);
    T ret = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, x.size()), (T)x[0], [&](tbb::blocked_range<size_t> r,T local_min) {
		for (size_t i = std::begin(r); i != std::end(r); i++)
		{
			local_min = std::min(local_min, x[i]);
		}
		return local_min;
		},
		[](T x,T y) {
			return std::min(x, y);
		});
    TOCK(minvalue);
    return ret;
}

template <class T>
std::vector<pod<T>> magicfilter(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(magicfilter);
    std::vector<pod<T>> res;
	res.resize(2*std::min(x.size(),y.size()));
	std::atomic<size_t> res_size = 0;

	tbb::parallel_for(tbb::blocked_range<size_t>(0, std::min(x.size(), y.size())),
		[&](tbb::blocked_range<size_t> r) {
			std::vector<pod<T>> local_res;
			local_res.resize(2 * r.size());
			size_t local_size = 0;
			for (size_t i = std::begin(r); i != std::end(r); i++)
			{
				if (x[i] > y[i]) {
					
					local_res[local_size++] = x[i];
				}
				else if (y[i] > x[i] && y[i] > 0.5f) {
					
					local_res[local_size++] = y[i];
					local_res[local_size++] = x[i] * y[i];
				}
			}
			size_t local_base = res_size.fetch_add(local_size);
			for (size_t i = 0; i < local_size; i++)
			{
				res[local_base + i] = local_res[i];
			}
		});
	res.resize(res_size);

	/*
	for (size_t i = 0; i < std::min(x.size(), y.size()); i++) {
        if (x[i] > y[i]) {
            res.push_back(x[i]);
        } else if (y[i] > x[i] && y[i] > 0.5f) {
            res.push_back(y[i]);
            res.push_back(x[i] * y[i]);
        }
    }
	*/
    
    TOCK(magicfilter);
    return res;
}

template <class T>
T scanner(std::vector<T> &x) {
    TICK(scanner);
    T ret = 0;
	ret = tbb::parallel_scan(tbb::blocked_range<size_t>(0, x.size()), T(0),
		[&](tbb::blocked_range<size_t> r,T local_res,auto is_final) {
			for (size_t i = std::begin(r); i != std::end(r); i++)
			{
				local_res += x[i];
				if (is_final)
				{
					x[i] = local_res;
				}
			}
			return local_res;
		}, [](T x,T y) {
			return x + y;
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
