#include <assert.h>
#include <cooperative_groups.h>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#define WINDOW_SIZE 32U
#define BLOCK_ROWS (WINDOW_SIZE >> 2)
#define TOTAL_WINDOW_SIZE (WINDOW_SIZE * WINDOW_SIZE)

#define SHARED_MEMORY_SIZE 1024U
#define WARP (SHARED_MEMORY_SIZE >> 1)

#define BATCH_SIZE 128U
#define NUM_REPS 1

inline void checkGPU(cudaError_t result, std::string file, int const line) {
  if (result != cudaSuccess) {
    fprintf(stderr, "Cuda Runtime Error at %s:%d : %s\n", file.c_str(), line,
            cudaGetErrorString(result));
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) checkGPU(val, __FILE__, __LINE__)

__device__ inline void Comparator(float &A, float &B, uint dir) {
  float t;
  // dir = 1;

  if ((A > B) == dir) {
    t = A;
    A = B;
    B = t;
  }
}

namespace cg = cooperative_groups;

__global__ void bitonicSortKernel(float *in, float *out, uint arrayLength,
                                  uint dir, uint batch, uint batchX, uint batchY, uint batchSize2D,
                                  uint noOfBatches) {

  cg::thread_block cta = cg::this_thread_block();
  __shared__ float shmem[TOTAL_WINDOW_SIZE];

  in += blockIdx.x * TOTAL_WINDOW_SIZE + threadIdx.x;

  shmem[threadIdx.x] = in[0];
  shmem[threadIdx.x + (TOTAL_WINDOW_SIZE / 2)] = in[(TOTAL_WINDOW_SIZE / 2)];

  // Bitonic merge
  for (uint size = 2; size < arrayLength; size <<= 1) {
    // Bitonic direction
    uint ddd = dir ^ ((threadIdx.x & (size / 2)) != 0);

    for (uint stride = size / 2; stride > 0; stride >>= 1) {
      cg::sync(cta);
      uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      Comparator(shmem[pos + 0], shmem[pos + stride], ddd);
    }
  }

  // ddd == dir for the last bitonic merge step
  {
    for (uint stride = arrayLength / 2; stride > 0; stride >>= 1) {
      cg::sync(cta);
      uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      Comparator(shmem[pos + 0], shmem[pos + stride], dir);
    }
  }

  cg::sync(cta);

  uint mid = TOTAL_WINDOW_SIZE / 2;
  float median = (shmem[mid-1] + shmem[mid]) / 2;

  // TODO: Stored value subtracted from median
  // in[0] = fabs(median - shmem[threadIdx.x]);
  // in[(TOTAL_WINDOW_SIZE / 2)] = fabs(median - shmem[threadIdx.x + (TOTAL_WINDOW_SIZE / 2)]);
  in[0] = shmem[threadIdx.x];
  in[(TOTAL_WINDOW_SIZE / 2)] = shmem[threadIdx.x + (TOTAL_WINDOW_SIZE / 2)];

  if (threadIdx.x == 0) {
    // Batching logic
    // uint idx = blockIdx.x;
    // out += (batchY * batchSize2D * WINDOW_SIZE) + batchX * batchSize2D;
    // out += (batchY * batchSize2D * WINDOW_SIZE) + batchX * batchSize2D;
    // uint yid = idx / batchSize2D;
    // uint xid = idx % batchSize2D;
    // uint off = (yid * WINDOW_SIZE) + xid;
    // printf("Reason %d xid %d yid %d blockIdx %d\n", off, xid, yid, blockIdx.x);
    out[blockIdx.x] = median;
  }
}

void bitonicSortBatches(float *median, float *madfm, float *arr, uint arrSize,
                        uint noOfWindows) {
  uint dir = 1;

  float *arr_d;
  float *median_d;
  float *madfm_d;

  checkCudaErrors(cudaMalloc(&arr_d, arrSize * sizeof(float)));
  checkCudaErrors(cudaMalloc(&median_d, noOfWindows * sizeof(float)));
  checkCudaErrors(cudaMalloc(&madfm_d, noOfWindows * sizeof(float)));

  checkCudaErrors(cudaMemcpyAsync(arr_d, arr, arrSize * sizeof(float),
                                  cudaMemcpyHostToDevice, 0));
  checkCudaErrors(cudaMemcpyAsync(median_d, median, noOfWindows * sizeof(float),
                                  cudaMemcpyHostToDevice, 0));
  checkCudaErrors(cudaMemcpyAsync(madfm_d, madfm, noOfWindows * sizeof(float),
                                  cudaMemcpyHostToDevice, 0));

  cudaEvent_t start, stop;

  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  checkCudaErrors(cudaEventRecord(start));

  uint batchSize = noOfWindows;
  uint threadCount = TOTAL_WINDOW_SIZE / 2;

  printf("blocks: %d threads: %d\n", batchSize, threadCount);

  
  bitonicSortKernel<<<batchSize, threadCount>>>(arr_d, median_d, TOTAL_WINDOW_SIZE, dir, 0, 0, 0, 0, 0);

  checkCudaErrors(cudaMemcpyAsync(median, median_d, noOfWindows * sizeof(float), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));

  checkCudaErrors(
      cudaMemcpy(arr, arr_d, arrSize * sizeof(float), cudaMemcpyDeviceToHost));

  float elapsedTime;
  checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("Num elements: %d Elapsed %.3fms Throughput: %.3f MElements/s \n", arrSize, elapsedTime,
         arrSize * 1e-6 / elapsedTime);

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

  checkCudaErrors(cudaFree(arr_d));
  checkCudaErrors(cudaFree(median_d));
  checkCudaErrors(cudaFree(madfm_d));
}

void printArray(float* arr, uint arrSize) {
  for (uint i = 0; i < arrSize; i++) {
    printf("%.1f ", arr[i]);
    if ((i + 1) % TOTAL_WINDOW_SIZE == 0) {
      printf("\n");
    }
  }
}

void printArrayMedian(float* arr, float* median, uint arrSize) {
  int mIdx = 0;
  for (uint i = 0; i < arrSize; i++) {
    printf("%.1f ", arr[i]);
    if ((i + 1) % TOTAL_WINDOW_SIZE == 0) {
      printf("\t median: %.1f \n", median[mIdx++]);
    }
  }
}

int main(int argc, char **argv) {

  int N = 64;

  if (argc > 1) {
    N = atoi(argv[1]);
  }

  int arrSize = TOTAL_WINDOW_SIZE * N;

  float *arr = (float *)malloc(sizeof(float) * arrSize);
  float *median = (float *)malloc(sizeof(float) * N);
  float *madfm = (float *)malloc(sizeof(float) * N);

  float range = 10.0f;
  for (int i = 0; i < arrSize; i++) {
    auto val = (double)std::rand() / (double)(RAND_MAX / range);
    arr[i] = val;
  }

  #ifdef DEBUG
  printf("Input:\n");
  printArray(arr, arrSize);
  #endif

  for (int i = 0; i < N; i++) {
    median[i] = 0.0f;
    madfm[i] = 0.0f;
  }

  bitonicSortBatches(median, madfm, arr, arrSize, N);

  #ifdef DEBUG
  printf("Output:\n");
  printArrayMedian(arr, median, arrSize);
  #endif

  free(madfm);
  free(median);
  free(arr);
}
