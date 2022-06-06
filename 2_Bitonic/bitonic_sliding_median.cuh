#ifndef BITONIC_SLIDING_MEDIAN_CUH
#define BITONIC_SLIDING_MEDIAN_CUH

#include <casacore/casa/Arrays.h>

void bitonic_sliding_median(float *median, float *madfm, float *arr,
                            unsigned int arrDim, unsigned int batchDim);

// Masked array to be implemented later
void bitonicSlidingSNR(casacore::Array<float> &median,
                       casacore::Array<float> &madfm,
                       casacore::Array<float> &array);

#endif
