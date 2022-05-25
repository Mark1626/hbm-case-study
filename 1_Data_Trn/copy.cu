#include <assert.h>
#include <cooperative_groups.h>
#include <stdio.h>
#include <stdlib.h>

namespace cg = cooperative_groups;

#define SHARED_SIZE_LIMIT 1024U
#define WARP 512U
#define WINDOW_SIZE SHARED_SIZE_LIMIT
const int NUM_REPS = 1000;

inline cudaError_t checkCudaErrors(cudaError_t result) {
  if (result != cudaSuccess) {
    fprintf(stderr, "Cuda Runtime Error : %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

__global__ void copy(float *out, float *in, uint size) {
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint stride = blockDim.x * gridDim.x;

  for (; tid < size; tid += stride) {
    out[tid] = in[tid];
  }
}

__global__ void copy_unroll(float *out, float *in, uint size) {
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint stride = blockDim.x * gridDim.x;

  #pragma unroll 4
  for (; tid < size; tid += stride) {
    out[tid] = in[tid];
  }
}

__global__ void copy2(float *out, float *in, uint size) {
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint stride = blockDim.x * gridDim.x;

  for (; tid < size/2; tid += stride) {
    reinterpret_cast<float2*>(out)[tid] = reinterpret_cast<float2*>(in)[tid];
  }
}

__global__ void copy4(float *out, float *in, uint size) {
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint stride = blockDim.x * gridDim.x;

  for (; tid < size/4; tid += stride) {
    reinterpret_cast<float4*>(out)[tid] = reinterpret_cast<float4*>(in)[tid];
  }
}

__global__ void copy4_unroll(float *out, float *in, uint size) {
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint stride = blockDim.x * gridDim.x;

  #pragma unroll 4
  for (; tid < size/4; tid += stride) {
    reinterpret_cast<float4*>(out)[tid] = reinterpret_cast<float4*>(in)[tid];
  }
}

void driver(uint size, int blocks, int threads,
            void (*kernel)(float *, float *, uint)) {

  float *in;
  float *out;
  cudaMallocManaged(&in, sizeof(float) * size);
  cudaMallocManaged(&out, sizeof(float) * size);

  for (uint i = 0; i < size; i++) {
    in[i] = (size - i) * 1.0;
  }

  cudaEvent_t start, stop;

  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  // Warmup
  kernel<<<blocks, threads>>>(out, in, size);

  checkCudaErrors(cudaEventRecord(start, 0));

  for (int i = 0; i < NUM_REPS; i++)
    kernel<<<blocks, threads>>>(out, in, size);

  cudaDeviceSynchronize();

#ifdef DEBUG
  for (int i = 0; i < size; i++) {
    printf("%f ", in[i]);
  }
  printf("\n");
#endif

  checkCudaErrors(cudaEventRecord(stop, 0));
  checkCudaErrors(cudaEventSynchronize(stop));

  float elapsedTime;
  checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("%.3f\t %.3fms\n", 2 * size * sizeof(float) * NUM_REPS * 1e-6 / elapsedTime, elapsedTime);
  // printf("Elapsed GPU time %f ms\n", elapsedTime);

#ifdef ASSERT
  for (int i = 0; i < size; i++) {
    if (out[i] != in[i]) {
      printf("Assertion failed at pos: %d\n", i);
      assert(out[i] == in[i]);
    }
  }
#endif

#ifdef DEBUG
  for (int i = 0; i < size; i++) {
    printf("%f ", out[i]);
  }
  printf("\n");
#endif

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

  cudaFree(out);
  cudaFree(in);
}

int main(int argc, char **argv) {
  int N;
  int blocks;
  int threads;

  if (argc < 4) {
    N = 1024;
    blocks = 256;
    threads = 256;
  } else {
    uint pow = atoi(argv[1]);
    N = 1 << pow;
    blocks = atoi(argv[2]);
    threads = atoi(argv[3]);
  }

  printf("Testing copy for %d elements\n", N);

  printf("Method \t\t\tBandwidth (GB/s) Time Taken\n");

  printf("Scalar copy\t\t\t");
  driver(N, blocks, threads, copy);

  printf("Scalar copy unroll\t\t");
  driver(N, blocks, threads, copy_unroll);

  printf("Vector float2 copy\t\t");
  driver(N, blocks, threads, copy2);

  printf("Vector float4 copy\t\t");
  driver(N, blocks, threads, copy4);

  printf("Vector float4 copy unroll\t");
  driver(N, blocks, threads, copy4_unroll);
}
