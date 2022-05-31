#include <assert.h>
#include <cstdlib>
#include <stdio.h>
#include <string>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define SHARED_SIZE_LIMIT 1024U
#define WARP 512U
#define WINDOW_SIZE SHARED_SIZE_LIMIT

#define NUM_REPS 1000

inline cudaError_t checkCudaErrors(std::string msg, cudaError_t result) {
  if (result != cudaSuccess) {
    fprintf(stderr, "Cuda Runtime Error %s : %s\n", msg.c_str(),
            cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

__global__ void copy(float *out, float *in, uint size) {
  uint tid = blockDim.x * blockIdx.x + threadIdx.x;
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

  for (; tid < size / 2; tid += stride) {
    reinterpret_cast<float2 *>(out)[tid] = reinterpret_cast<float2 *>(in)[tid];
  }
}

__global__ void copy4(float *out, float *in, uint size) {
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint stride = blockDim.x * gridDim.x;

  for (; tid < size / 4; tid += stride) {
    reinterpret_cast<float4 *>(out)[tid] = reinterpret_cast<float4 *>(in)[tid];
  }
}

__global__ void copy4_unroll(float *out, float *in, uint size) {
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint stride = blockDim.x * gridDim.x;

#pragma unroll 4
  for (; tid < size / 4; tid += stride) {
    reinterpret_cast<float4 *>(out)[tid] = reinterpret_cast<float4 *>(in)[tid];
  }
}

void driver(int size, int blocks, int threads, void (*kernel)(float*, float*, uint)) {
  float *in;
  float *out;

  in = (float *)malloc(size * sizeof(float));
  out = (float *)malloc(size * sizeof(float));

  float *in_d;
  float *out_d;

  cudaEvent_t start, stop;

  checkCudaErrors("Start Event Create", cudaEventCreate(&start));
  checkCudaErrors("Stop Event Create", cudaEventCreate(&stop));

  checkCudaErrors("Malloc", cudaMalloc(&in_d, size * sizeof(float)));
  checkCudaErrors("Malloc", cudaMalloc(&out_d, size * sizeof(float)));

  for (int i = 0; i < size; i++) {
    in[i] = size - i;
  }

  checkCudaErrors("HtoD", cudaMemcpy(in_d, in, size * sizeof(float),
                                     cudaMemcpyHostToDevice));

  kernel<<<blocks, threads>>>(out_d, in_d, size);

  checkCudaErrors("Start", cudaEventRecord(start, 0));

  for (int i = 0; i < NUM_REPS; i++)
    kernel<<<blocks, threads>>>(out_d, in_d, size);

  checkCudaErrors("Stop", cudaEventRecord(stop, 0));
  checkCudaErrors("Sync", cudaEventSynchronize(stop));

  float elapsedTime;
  checkCudaErrors("Time taken",
                  cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("%.3fms\t %.3f\n", elapsedTime,
         2 * size * sizeof(float) * NUM_REPS * 1e-6 / elapsedTime);

  checkCudaErrors("Event destroy start", cudaEventDestroy(start));
  checkCudaErrors("Event destroy stop", cudaEventDestroy(stop));

  checkCudaErrors("DtoH", cudaMemcpy(out, out_d, size * sizeof(float),
                                     cudaMemcpyDeviceToHost));

  free(in);
  free(out);
  checkCudaErrors("cuda free", cudaFree(in_d));
  checkCudaErrors("cuda free", cudaFree(out_d));
}

int main(int argc, char **argv) {
  int N = 1 << 20;
  int blocks = 256;
  int threads = 256;

  if (argc > 1) {
    int pow = atoi(argv[1]);
    N = 1 << pow;
  }

  printf("Method\t\t\t Time Taken Bandwidth (GB/s)\n");
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
