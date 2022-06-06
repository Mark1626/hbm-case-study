#include <algorithm>
#include <assert.h>
#include <cooperative_groups.h>
#include <stdio.h>

#include "bitonic_sliding_median.cuh"
#include "config.h"

// #define DEBUG
// #define INFO

static inline cudaError_t checkCudaErrors(cudaError_t result) {
  if (result != cudaSuccess) {
    fprintf(stderr, "Cuda Runtime Error : %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

__device__ inline void Comparator(float &A, float &B, uint dir) {
  float t;
  // dir = 1;

  if ((A > B) == dir) {
    t = A;
    A = B;
    B = t;
  }
}

// Bitonic sort kernel based on Cuda's example for arrays fitting into shared
// memory
__global__ void bitonicSortKernel(float *in, float *out, uint arrayLength,
                                  uint dir, uint batch, uint batchSize, uint batchX, uint batchY, uint batchSize2D, uint wdim,
                                  uint noOfBatches) {

  if (blockIdx.x < noOfBatches) {
    // Handle to thread block group
    cooperative_groups::thread_block cta =
        cooperative_groups::this_thread_block();
    // Shared memory storage for one or more short vectors
    __shared__ float shmem[TOTAL_WINDOW_SIZE];

    // Offset to the beginning of subbatch and load data
    in += blockIdx.x * TOTAL_WINDOW_SIZE + threadIdx.x;

    shmem[threadIdx.x + 0] = in[0];
    shmem[threadIdx.x + (TOTAL_WINDOW_SIZE / 2)] = in[(TOTAL_WINDOW_SIZE / 2)];

    for (uint size = 2; size < arrayLength; size <<= 1) {
      // Bitonic direction
      uint ddd = dir ^ ((threadIdx.x & (size / 2)) != 0);

      for (uint stride = size / 2; stride > 0; stride >>= 1) {
        cooperative_groups::sync(cta);
        uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        Comparator(shmem[pos + 0], shmem[pos + stride], ddd);
      }
    }

    // ddd == dir for the last bitonic merge step
    {
      for (uint stride = arrayLength / 2; stride > 0; stride >>= 1) {
        cooperative_groups::sync(cta);
        uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        Comparator(shmem[pos + 0], shmem[pos + stride], dir);
      }
    }

    cooperative_groups::sync(cta);

    uint mid = TOTAL_WINDOW_SIZE / 2;
    float median = (shmem[mid - 1] + shmem[mid]) / 2;

    
    // TODO: Is doing this in one thread efficient?
    
    if (threadIdx.x == 0) {
      // printf("batch %d batchX %d batchY %d\n", batch, batchX, batchY);
      out += (batchY * batchSize2D * wdim) + batchX * batchSize2D;

      uint idx = blockIdx.x;
      uint yid = idx / batchSize2D;
      uint xid = idx % batchSize2D;
      uint off = (yid * (wdim)) + xid;
      // printf("Reason %d xid %d yid %d blockIdx %d\n", off, xid, yid, blockIdx.x);
      out[off] = median;
      // out[blockIdx.x + (batch * batchSize)] = median;
    }

    in[0] = fabs(median - shmem[threadIdx.x + 0]);
    in[(TOTAL_WINDOW_SIZE / 2)] =
        fabs(median - shmem[threadIdx.x + (TOTAL_WINDOW_SIZE / 2)]);
  }
}

__global__ void load_2d_kernel(float *windows, float *in, uint dim, uint batchX,
                               uint batchY, uint batchSize, uint noOfBatches) {
  // This is a square so we skip checking Y
  if (blockIdx.x < noOfBatches) {
    uint batchOffsetX = batchX * batchSize;
    uint batchOffsetY = batchY * batchSize;
    uint bx = blockIdx.x;
    uint by = blockIdx.y;
    int offset = (by * batchSize + bx) * TOTAL_WINDOW_SIZE;

    windows += offset;

    int x = threadIdx.x;
    int y = threadIdx.y;

    int ax = (bx) + batchOffsetX + threadIdx.x;
    int ay = (by) + batchOffsetY + threadIdx.y;

    for (uint j = 0; j < WINDOW_SIZE; j += BLOCK_ROWS) {
      windows[(y + j) * WINDOW_SIZE + x] = in[(ay + j) * dim + ax];
    }
  }
}

void bitonic_sliding_median(float *median, float *madfm, float *arr,
                            unsigned int arrayDim, unsigned int batchDim) {
  uint dir = 1;

  uint arraySize = arrayDim * arrayDim;

  uint wdim = (arrayDim - WINDOW_SIZE);
  uint noOfWindows = wdim * wdim;

  batchDim = std::min(wdim, batchDim);
  uint batchSize = batchDim * batchDim;
  dim3 batchDim2D(batchDim, batchDim);

  uint bufferSize = batchSize * TOTAL_WINDOW_SIZE;

  float *d_arr;
  float *d_median;
  float *d_madfm;
  float *d_windows;
  float *h_windows;

#ifdef DEBUG
  {
    // printf("Input: \n");
    // for (uint i = 0; i < arraySize; i++) {
    //   printf("%.1f ", arr[i]);
    //   if ((i + 1) % arrayDim == 0) {
    //     printf("\n");
    //   }
    // }
    // printf("\n");
  }

#endif

#ifdef INFO
  printf("Array: %d %d size: %d\n", arrayDim, arrayDim, arraySize);
  printf("wdim %d %d\n", wdim, wdim);
  printf("buffer elem: %d buffer size %.2fkb\n", bufferSize,
         (4.0f * bufferSize) / 1024.0f);
#endif

  h_windows = (float *)malloc(bufferSize * sizeof(float));

  checkCudaErrors(cudaMalloc(&d_arr, arraySize * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_median, noOfWindows * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_madfm, noOfWindows * sizeof(float)));
  checkCudaErrors(cudaMemcpyAsync(d_arr, arr, arraySize * sizeof(float),
                                  cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&d_windows, bufferSize * sizeof(float)));

  cudaEvent_t start, stop;

  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  checkCudaErrors(cudaEventRecord(start));

  
  uint batch2Dim = wdim / batchDim;
  uint blocks = noOfWindows;
  uint noOfBatches = blocks/ batchSize;
  
  uint threadCount = (TOTAL_WINDOW_SIZE / 2);
  dim3 threadCount2D(WINDOW_SIZE, BLOCK_ROWS);

#ifdef INFO
  printf("BatchDim: %d %d %d\n", batchDim2D.x, batchDim2D.y, batchDim2D.z);
  printf("Threads: %d %d %d\n", threadCount2D.x, threadCount2D.y,
         threadCount2D.z);
  printf("BatchSize: %u Windows: %u\n", batchSize, noOfWindows);
  printf("Populating for %d batches\n", noOfBatches);
#endif

  uint noOfBlocks = batchSize;
  uint noOfBlocks2D = batchDim;

  for (uint batch = 0; batch < noOfBatches; batch++) {
  // for (uint batch = 1; batch < 2; batch++) {
    // TODO: Implement non power of 2 array logic
    // if (blocks % batchSize != 0 && batch == noOfBatches-1) {
    //   noOfBlocks = blocks % batchSize;
    //   noOfBlocks2D = wdim % batchDim;
    // }

    uint batchX = batch % batch2Dim;
    uint batchY = batch / batch2Dim;
#ifdef DEBUG
    printf("Batch: %d batchx %d batchy %d\n", batch, batchX, batchY);
    printf("blocks: %d blocks2d %d\n", noOfBlocks, noOfBlocks2D);
#endif
    load_2d_kernel<<<batchDim2D, threadCount2D>>>(
        d_windows, d_arr, arrayDim, batchX, batchY, batchDim, noOfBlocks2D);

#ifdef DEBUG
    {
      int row = 1;
      checkCudaErrors(cudaMemcpy(h_windows, d_windows,
                                 bufferSize * sizeof(float),
                                 cudaMemcpyDeviceToHost));
      printf("\nOutput Before Sort: batch%d\n", batch);
      int idx = 0;
      for (uint window = 0; window < batchSize; window++) {
        printf("[");
        for (uint i = 0; i < TOTAL_WINDOW_SIZE; i++) {
          printf("%d, ", (int)h_windows[idx]);
          idx++;
          if ((i + 1) % WINDOW_SIZE == 0) {
            printf("\n ");
          }
        }
        printf("]");
        printf("\nRow%d.\n", row++);
      }
    }
#endif

    bitonicSortKernel<<<batchSize, threadCount>>>(d_windows, d_median,
                                                  TOTAL_WINDOW_SIZE, dir, batch,
                                                  batchSize, batchX, batchY, batchDim2D.x, wdim, noOfBlocks);

    bitonicSortKernel<<<batchSize, threadCount>>>(d_windows, d_madfm,
                                                  TOTAL_WINDOW_SIZE, dir, batch,
                                                  batchSize, batchX, batchY, batchDim2D.x, wdim, noOfBlocks);
  }

  checkCudaErrors(cudaMemcpyAsync(median, d_median, sizeof(float) * noOfWindows,
                                  cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaMemcpyAsync(madfm, d_madfm, sizeof(float) * noOfWindows,
                                  cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));

  float elapsedTime;
  checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("Elapsed GPU time %f ms\n", elapsedTime);

  #ifdef DEBUG
    {
      int row = 1;
      checkCudaErrors(cudaMemcpy(h_windows, d_windows, bufferSize *
      sizeof(float),
                                 cudaMemcpyDeviceToHost));
      printf("\nOutput After Sort:\n");
      int idx = 0;
      for (uint window = 0; window < batchSize; window++) {
        for (uint i = 0; i < TOTAL_WINDOW_SIZE; i++) {
          printf("%.1f ", h_windows[idx]);
          idx++;
          if ((i + 1) % WINDOW_SIZE == 0) {
            printf("\n");
          }
        }
        printf("\nRow%d.\n", row);
      }
    }
  #endif

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

  free(h_windows);
  checkCudaErrors(cudaFree(d_median));
  checkCudaErrors(cudaFree(d_madfm));
  checkCudaErrors(cudaFree(d_windows));
  checkCudaErrors(cudaFree(d_arr));
}
