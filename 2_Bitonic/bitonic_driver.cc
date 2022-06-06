#include "bitonic_sliding_median.cuh"
#include "config.h"
#include <stdio.h>
#include <assert.h>
#include <casacore/casa/Arrays.h>

static unsigned int factorRadix2(unsigned int *log2L, unsigned int L) {
  if (!L) {
    *log2L = 0;
    return 0;
  } else {
    for (*log2L = 0; (L & 1) == 0; L >>= 1, *log2L++)
      ;

    return L;
  }
}

void bitonicSlidingSNR(casacore::Array<float> &median,
                       casacore::Array<float> &madfm,
                       casacore::Array<float> &array) {

  casacore::IPosition hboxsz(2, WINDOW_SIZE / 2, WINDOW_SIZE / 2);
  // Assert present of contiguousStorage
  assert(median.contiguousStorage());
  assert(madfm.contiguousStorage());
  assert(array.contiguousStorage());

  const casacore::IPosition &shape = array.shape();

  // TODO: Assert size of array is WINDOW_SIZE + 2^N

  // Assert array dimensions is 2
  // TODO: This should be increased to support MxNx1x1
  // Assert this is a square matrix
  assert(shape.size() == 2);
  assert(shape[0] == shape[1]);

  size_t arrDim = shape[0];

  assert(arrDim - WINDOW_SIZE > 0);

  unsigned int power;
  unsigned int rem = factorRadix2(&power, arrDim - WINDOW_SIZE);

  if (rem != 1) {
    fprintf(stderr, "Size of array must be a ((power of 2 + %d)\n", WINDOW_SIZE);
    assert(rem == 1);
  }

  size_t ndim = array.ndim();
  casacore::IPosition resShape(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    resShape[i] = shape[i] - WINDOW_SIZE;
  }

  casacore::Array<float> medianRes(resShape);
  casacore::Array<float> madfmRes(resShape);

  float *medianData = medianRes.data();
  float *madfmData = madfmRes.data();
  float *arrData = array.data();

  bitonic_sliding_median(medianData, madfmData, arrData, arrDim, BATCH_DIM);

  casacore::Array<float> fullMedianRes(shape);
  casacore::Array<float> fullMadfmRes(shape);

  fullMedianRes = float();
  fullMadfmRes = float();

  fullMedianRes(hboxsz, resShape + hboxsz - 1) = medianRes;
  fullMadfmRes(hboxsz, resShape + hboxsz - 1) = madfmRes;

  median = std::move(fullMedianRes);
  madfm = std::move(fullMadfmRes);
}
