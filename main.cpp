#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "ticktock.h"
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_scan.h>
#include <mutex>
#include "pod.h"


// TODO: 并行化所有这些 for 循环

template <class T, class Func>
std::vector<T> fill(std::vector<T> &arr, Func const &func) {
    TICK(fill);
	size_t n = arr.size();
	tbb::parallel_for(tbb::blocked_range<size_t>(0, n), 
	[&](tbb::blocked_range<size_t> r) {
		for (size_t i = r.begin(); i < r.end(); i ++) {
			arr[i] = func(i);
		}
	});
    TOCK(fill);
    return arr;
}

template <class T>
void saxpy(T a, std::vector<pod<T>> &x, std::vector<pod<T>> const &y) {
    TICK(saxpy);
	size_t n = x.size();
	tbb:parallel_for(tbb::blocked_range<size_t>(0, n),
	[&](tbb::blocked_range<size_t> r) {
		for (size_t i = r.begin(); i < r.end(); i ++) {
			x[i] = a * x[i] + y[i];
		}
	}); 
    TOCK(saxpy);
}

template <class T>
T sqrtdot(std::vector<pod<T>> const &x, std::vector<pod<T>> const &y) {
    TICK(sqrtdot);
	size_t n = std::min(x.size(), y.size());
    T res = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, n), (float)0,
	[&](tbb::blocked_range<size_t> r, T local_res) {
		for (size_t i = r.begin(); i < r.end(); i ++) {
			local_res += x[i] * y[i];
		}
		return local_res;
	}, [](float x, float y) {
		return x + y;
	});
    res = std::sqrt(res);
    TOCK(sqrtdot);
    return res;
}

template <class T>
T minvalue(std::vector<pod<T>> const &x) {
    TICK(minvalue);
	size_t n = x.size();
	T res = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, n), x[0], 
	[&](tbb::blocked_range<size_t> r, auto local_res) {
		for (size_t i = r.begin(); i < r.end(); i ++) {
			local_res = std::min(local_res, x[i]);
		}
		return local_res;
	}, [](T x, T y) {
		return std::min(x, y);
	});
    TOCK(minvalue);
    return res;
}

template <class T>
std::vector<pod<T>> magicfilter(std::vector<pod<T>> const &x, std::vector<pod<T>> const &y) {
    TICK(magicfilter);
    std::vector<pod<T>> res;
	std::mutex mtx;
	size_t n = std::min(x.size(), y.size());
	
	res.reserve(n);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
	[&](tbb::blocked_range<size_t> r){
		std::vector<float> local_res;
		local_res.reserve(r.size());
		for (size_t i = r.begin(); i < r.end(); i ++) {
			if (x[i] > y[i]) {
				local_res.push_back(x[i]);
			} else if (y[i] > x[i] && y[i] > 0.5f) {
				local_res.push_back(y[i]);
				local_res.push_back(x[i] * y[i]);
			}
		}
		std::lock_guard lck(mtx);
		std::copy(local_res.begin(), local_res.end(), std::back_inserter(res));
	});
    TOCK(magicfilter);
    return res;
}

template <class T>
T scanner(std::vector<pod<T>> &x) {
    TICK(scanner);
	size_t n = x.size();
	T res = tbb::parallel_scan(tbb::blocked_range<size_t>(0, n), (float)0, 
	[&](tbb::blocked_range<size_t> r, T local_res, auto is_final) {
		for (size_t i = r.begin(); i < r.end(); i ++) {
			local_res += x[i];
			if (is_final) {
				x[i] = local_res;
			}
		}
		return local_res;
	}, [](T x, T y) {
		return x + y;	
	});
    TOCK(scanner);
    return res;
}

int main() {
    size_t n = 1<<26;
    std::vector<pod<float>> x(n);
    std::vector<pod<float>> y(n);

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
