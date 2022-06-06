# Bitonic Sliding Window Median

## Building

```
mkdir -p build && cd build
CXX=$(hipconfig -l)/clang++ cmake .. -DUSE_CUDA=OFF -DUSE_HIP=ON
make

mkdir -p build && cd build
cmake .. -DUSE_CUDA=ON
make
```
