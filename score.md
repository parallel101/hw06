# 原版

fill: 0.728694s
fill: 0.74406s
saxpy: 0.0409412s
sqrtdot: 0.0724723s
5165.4
minvalue: 0.0691992s
-1.11803
magicfilter: 0.371052s
55924034
scanner: 0.0702286s
6.18926e+07

# 2

fill: 0.110683s
fill: 0.10929s
saxpy: 0.0116205s
sqrtdot: 0.011483s
5792.62
minvalue: 0.00942511s
-1.11803
magicfilter: 0.0292501s
55924034
scanner: 0.0187589s
6.19048e+07

# 3
只是针对parallel scan，发现手动划分task并行比parallel_scan更快，应该是我没找到最佳的partitioner，但是我测试了好几种都是手动划分更快
fill: 0.111643s
fill: 0.113284s
saxpy: 0.0120721s
sqrtdot: 0.0143203s
5792.62
minvalue: 0.00966027s
-1.11803
magicfilter: 0.0297295s
55924034
scanner: 0.016231s
6.19332e+07