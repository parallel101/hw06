# 原版
fill: 1.66595s
fill: 1.65791s
saxpy: 0.0670104s
sqrtdot: 0.0785495s
5165.4
minvalue: 0.0702195s
-1.11803
magicfilter: 0.466389s
55924034
scanner: 0.0654257s
6.18926e+07

# fill,saxpy,minvalue -> OMP
'''c++

template <class T, class Func>
std::vector<T> fill(std::vector<T> &arr, Func const &func) {
    TICK(fill);
    size_t len = arr.size();
    #pragma omp parallel for
    for (size_t i = 0; i < len; i++) {
        arr[i] = func(i);
    }
    TOCK(fill);
    return arr;
}

template <class T>
void saxpy(T a, std::vector<T> &x, std::vector<T> const &y) {
    TICK(saxpy);
    #pragma omp parallel for
    for (size_t i = 0; i < x.size(); i++) {
       x[i] = a * x[i] + y[i];
    }
    TOCK(saxpy);
}

template <class T>
T sqrtdot(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(sqrtdot);
    T ret = 0;
    #pragma omp parallel for
    for (size_t i = 0; i < std::min(x.size(), y.size()); i++) {
        ret += x[i] * y[i];
    }
    ret = std::sqrt(ret);
    TOCK(sqrtdot);
    return ret;
}

template <class T>
T minvalue(std::vector<T> const &x) {
    TICK(minvalue);
    T ret = x[0];
    #pragma omp parallel for
    for (size_t i = 1; i < x.size(); i++) {
        if (x[i] < ret)
            ret = x[i];
    }
    TOCK(minvalue);
    return ret;
}
'''

fill: 0.0781128s
fill: 0.0733396s
saxpy: 0.0209165s
minvalue: 0.0081233s
-1.11803



# sqrtdot -> tbb::parallel_reduce
'''c++

template <class T>
T sqrtdot(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(sqrtdot);
    size_t len = std::min(x.size(), y.size());
    T ret = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, len), T{},
    [&](tbb::blocked_range<size_t> r, T local_res){
        for(size_t i = r.begin(); i<r.end(); i++){
            local_res += x[i] * y[i];
        }
        return local_res;
    }, [](T x, T y){
        return x + y;
    });

/*
    T ret = 0;
    for (size_t i = 0; i < std::min(x.size(), y.size()); i++) {
        ret += x[i] * y[i];
    }
*/
    ret = std::sqrt(ret);
    TOCK(sqrtdot);
    return ret;
}
'''

sqrtdot: 0.0120102s


# magicfilter -> parallel_for

'''c++

template <class T>
std::vector<T> magicfilter(std::vector<T> const &x, std::vector<T> const &y) {
    TICK(magicfilter);
    std::vector<T> res;

    size_t n = std::min(x.size(), y.size());
    res.reserve(n*3);
    tbb::spin_mutex mtx;
    tbb::task_arena ta(20);
    ta.execute([&]{
            tbb::parallel_for(tbb::blocked_range<size_t>(0, n), 
            [&](tbb::blocked_range<size_t> r){
                std::vector<T> local_res;
                local_res.reserve(r.size()*2);
                for(size_t i=r.begin(); i<r.end(); i++){
                    if (x[i] > y[i]) {
                        local_res.push_back(x[i]);
                    } else if (y[i] > x[i] && y[i] > 0.5f) {
                        local_res.push_back(y[i]);
                        local_res.push_back(x[i] * y[i]);
                    }
                }
                std::lock_guard<tbb::spin_mutex> lock_guard(mtx);
                std::copy(local_res.begin(), local_res.end(), std::back_inserter(res));
            }, tbb::auto_partitioner{});
    }
    );


    TOCK(magicfilter);
    return res;
}

'''

magicfilter: 0.270902s

使用simple_paritioner会更慢，要20s。

# scanner -> tbb::parallel_scan
scanner: 0.0147832s