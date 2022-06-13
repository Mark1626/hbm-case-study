#include <cstdlib>
#include <limits>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "colors.h"

#define SHMEM_WINDOW_SIZE 32U
#define SHARED_MEMORY_SIZE (SHMEM_WINDOW_SIZE * SHMEM_WINDOW_SIZE)
#define SHMEM_HALF (SHARED_MEMORY_SIZE >> 1)

#define NUM_ITER 1
// #define ASSERT

inline void checkGPU(cudaError_t result, std::string file, int const line) {
  if (result != cudaSuccess) {
    fprintf(stderr, "Cuda Runtime Error at %s:%d : %s\n", file.c_str(), line,
            cudaGetErrorString(result));
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) checkGPU(val, __FILE__, __LINE__)

void printArray(float *arr, uint arrSize, uint wSize) {
  for (uint i = 0; i < arrSize; i++) {
    printf("%.2f ", arr[i]);
    if ((i + 1) % wSize == 0) {
      printf("\n");
    }
  }
}

void printArrayMedian(float *arr, float *median, uint arrSize, uint wSize) {
  int mIdx = 0;

  int rows = arrSize / wSize;
  for (uint row = 0; row < rows; row++) {
    float prev = arr[row * wSize];
    printf("%.0f ", prev);
    for (uint cell = 1; cell < wSize; cell++) {
      float val = arr[row * wSize + cell];
      if (prev > val) {
        printf(C_BGRED);
      }
      prev = val;
      printf("%.0f ", val);
    }
    printf(C_RESET);
    printf("\t median: " BLUE_TEXT("%.1f\n"), median[mIdx++]);
  }
}

uint highest_power_2(uint n) {
  uint t = n - 1;
  t |= t >> 1;
  t |= t >> 2;
  t |= t >> 4;
  t |= t >> 8;
  t |= t >> 16;
  return t + 1;
}

__global__ void transform(float *in, float *out, uint arrSize, uint real_wsize,
                          uint wsize) {

  in += blockIdx.x * real_wsize;

  uint mid = wsize / 2;
  float median = in[mid];
  median += (wsize & 1) ? in[mid] : in[mid - 1];
  median /= 2;

  in += threadIdx.x;

  for (int stride = 0; stride < real_wsize; stride += SHARED_MEMORY_SIZE) {
    in[stride] = fabs(median - in[stride]);
    in[stride + (SHARED_MEMORY_SIZE / 2)] =
        fabs(median - in[stride + (SHARED_MEMORY_SIZE / 2)]);
  }

  if (threadIdx.x == 0) {
    out[blockIdx.x] = median;
  }
}

__global__ void transform_shmem(float *in, float *out, uint arrSize,
                                uint real_wsize, uint wsize) {

  in += blockIdx.x * real_wsize;

  __shared__ float shmem[SHARED_MEMORY_SIZE];

  uint mid = wsize / 2;
  float median = in[mid];
  median += (wsize & 1) ? in[mid] : in[mid - 1];
  median /= 2;

  in +=  threadIdx.x;

  for (int stride = 0; stride < real_wsize; stride += SHARED_MEMORY_SIZE) {
    shmem[threadIdx.x] = in[stride];
    shmem[threadIdx.x + (SHARED_MEMORY_SIZE / 2)] =
        in[stride + (SHARED_MEMORY_SIZE / 2)];

    in[stride] = fabs(median - shmem[threadIdx.x]);
    in[stride + (SHARED_MEMORY_SIZE / 2)] =
        fabs(median - shmem[threadIdx.x + (SHARED_MEMORY_SIZE / 2)]);
  }

  if (threadIdx.x == 0) {
    out[blockIdx.x] = median;
  }
}

__global__ void transform_shmem_2(float *in, float *out, uint arrSize,
                                uint real_wsize, uint wsize) {

  in += blockIdx.x * real_wsize;

  __shared__ float shmem[SHARED_MEMORY_SIZE];

  uint mid = wsize / 2;
  float median = in[mid];
  median += (wsize & 1) ? in[mid] : in[mid - 1];
  median /= 2;

  in +=  threadIdx.x;

  for (int stride = 0; stride < real_wsize; stride += SHARED_MEMORY_SIZE) {
    shmem[threadIdx.x] = in[stride];
    shmem[threadIdx.x + (SHARED_MEMORY_SIZE / 2)] =
        in[stride + (SHARED_MEMORY_SIZE / 2)];
  }

  for (int stride = 0; stride < real_wsize; stride += SHARED_MEMORY_SIZE) {
    in[stride] = fabs(median - shmem[threadIdx.x]);
    in[stride + (SHARED_MEMORY_SIZE / 2)] =
        fabs(median - shmem[threadIdx.x + (SHARED_MEMORY_SIZE / 2)]);
  }

  if (threadIdx.x == 0) {
    out[blockIdx.x] = median;
  }
}


static int compfloat(const void *p1, const void *p2) {
  float *v1 = (float *)p1;
  float *v2 = (float *)p2;
  if (*v1 < *v2)
    return -1;
  if (*v1 > *v2)
    return 1;
  return 0;
}

#define EPHISILON 0.001f

void assertArray(std::string str, float *exp, float *act, uint N) {
  int assert_failure = 0;
  for (int i = 0; i < N; i++) {
    if (abs(exp[i] - act[i]) > EPHISILON) {
      printf("Assertion failure: %s " RED_TEXT(" exp: %f actual: %f\n"),
             str.c_str(), exp[i], act[i]);
      assert_failure = 1;
    }
  }
  if (!assert_failure)
    printf(GREEN_TEXT("Assertion passed %s\n"), str.c_str());
}

void calculate_assertion_values(int N, float *expect_arr, float *expect_median,
                                uint real_wsize, uint wsize) {
  // Calculate Median
  for (uint window = 0; window < N; window++) {
    float *win_ptr = expect_arr + (window * real_wsize);
    std::qsort(win_ptr, real_wsize, sizeof(float), compfloat);
    int mid = wsize / 2;
    if (wsize & 1) {
      expect_median[window] = win_ptr[mid];
    } else {
      expect_median[window] = (win_ptr[mid] + win_ptr[mid - 1]) / 2;
    }

    // Subtract median
    for (uint cell = 0; cell < real_wsize; cell++) {
      if (cell < wsize) {
        expect_arr[window * real_wsize + cell] = fabs(
            expect_arr[window * real_wsize + cell] - expect_median[window]);
      }
    }
  }
}

void driver(uint arrSize, uint N, uint real_wsize, uint wsize,
            void kernel(float *, float *, uint, uint, uint)) {
  float *arr = (float *)malloc(arrSize * sizeof(float));
  float *median = (float *)malloc(N * sizeof(float));

  float *expect_arr = (float *)malloc(arrSize * sizeof(float));
  float *expect_median = (float *)malloc(N * sizeof(float));

  for (uint row = 0; row < N; row++) {
    for (uint cell = 0; cell < real_wsize; cell++) {
      if (cell < wsize) {
        arr[row * real_wsize + cell] = (row * real_wsize + cell) * 1.0f;
      } else {
        arr[row * real_wsize + cell] = std::numeric_limits<float>::infinity();
      }
    }
  }

  for (int i = 0; i < arrSize; i++) {
    expect_arr[i] = arr[i];
  }

  calculate_assertion_values(N, expect_arr, expect_median, real_wsize, wsize);

  uint blocks = N;
  uint threads = SHARED_MEMORY_SIZE / 2;

  float *arr_d;
  float *median_d;

#ifdef DEBUG
  printArray(arr, arrSize, real_wsize);
#endif

  checkCudaErrors(cudaMalloc(&arr_d, arrSize * sizeof(float)));
  checkCudaErrors(cudaMalloc(&median_d, real_wsize * sizeof(float)));

  checkCudaErrors(cudaMemcpyAsync(arr_d, arr, arrSize * sizeof(float),
                                  cudaMemcpyHostToDevice, 0));

  cudaEvent_t start, stop;

  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  // Warm up
  // transform<<<blocks, threads>>>(arr_d, median_d, arrSize, real_wsize,
  // wsize);

  checkCudaErrors(cudaEventRecord(start));

  kernel<<<blocks, threads>>>(arr_d, median_d, arrSize, real_wsize, wsize);

  // for (int i = 1; i < NUM_ITER; i++) {
  //   transform<<<blocks, threads>>>(arr_d, median_d, arrSize, real_wsize,
  //   wsize);
  // }

  checkCudaErrors(cudaMemcpyAsync(median, median_d, blocks * sizeof(float),
                                  cudaMemcpyDeviceToHost));

  float elapsedTime;

  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

  checkCudaErrors(
      cudaMemcpy(arr, arr_d, arrSize * sizeof(float), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

#ifdef ASSERT
  // assertWindows(arr, arrSize, real_wsize, wsize);
  assertArray("median", median, expect_median, N);
  for (int row = 0; row < N; row++) {
    assertArray("Row " + std::to_string(row), expect_arr + row * real_wsize,
                arr + row * real_wsize, real_wsize);
  }
#endif

  printf("Elements: " BLUE_TEXT(" %d ") " Time: " BLUE_TEXT(
             "%fms") " Throughput: " BLUE_TEXT("%f MElem/sec") " BandWidth:"
                                                               " " BLUE_TEXT(
                                                                   "%fGB"
                                                                   "/s") "\n",
         arrSize, elapsedTime,
         (1e-6 * arrSize * NUM_ITER) / (1e-3 * elapsedTime),
         ((2 * arrSize * 4 * 1e-6 * NUM_ITER) / elapsedTime));

#ifdef DEBUG
  printArray(arr, arrSize, real_wsize);
#endif

  checkCudaErrors(cudaFree(arr_d));
  checkCudaErrors(cudaFree(median_d));

  free(arr);
  free(median);
}

int main(int argc, char **argv) {
  uint N = 128;
  uint wdim = 128;
  // uint N = 4;
  // uint wdim = 4;
  if (argc > 2) {
    N = atoi(argv[1]);
    wdim = atoi(argv[2]);
  }
  uint wsize = wdim * wdim;
  uint real_wsize = highest_power_2(wsize);
  uint arrSize = N * real_wsize;

  cudaDeviceSynchronize();

  printf("Normal approach\n");
  driver(arrSize, N, real_wsize, wsize, transform);

  printf("With Shared mem\n");
  driver(arrSize, N, real_wsize, wsize, transform_shmem);

  printf("With Shared mem 2\n");
  driver(arrSize, N, real_wsize, wsize, transform_shmem_2);
}
