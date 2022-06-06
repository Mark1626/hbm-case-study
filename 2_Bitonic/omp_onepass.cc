#pragma once

#include "config.h"

#include <assert.h>
#include <casacore/casa/Arrays/Array.h>
#include <casacore/casa/Arrays/IPosition.h>

namespace casa = casacore;

#define THRESHOLD 0.01f

class StatFunc {
public:
  explicit StatFunc(bool sorted = false, bool takeEvenMean = true)
      : itsSorted(sorted), itsTakeEvenMean(takeEvenMean) {}
  void operator()(const casacore::Array<float> &arr, float &median,
                  float &madfm) const {
    // std::cout << arr << std::endl;
    median = casa::median(arr, itsSorted, itsTakeEvenMean);
    // Inlining the MADFM is the actual time save
    casa::Array<float> absdiff = abs(arr - median);
    madfm = casa::median(absdiff, false, itsTakeEvenMean);
  }

private:
  bool itsSorted;
  bool itsTakeEvenMean;
  bool itsInPlace;
};

template <typename DualStatFunc>
void onePassSlidingArrayMath(casa::Array<float> &stat1,
                             casa::Array<float> &stat2,
                             const casa::Array<float> &array,
                             const casa::IPosition &halfBoxSize,
                             const DualStatFunc &funcObj, bool fillEdge) {
  size_t ndim = array.ndim();
  const casa::IPosition &shape = array.shape();
  // Set full box size (-1) and resize/fill as needed.
  casa::IPosition hboxsz(2 * halfBoxSize);
  if (hboxsz.size() != array.ndim()) {
    size_t sz = hboxsz.size();
    hboxsz.resize(array.ndim());
    for (size_t i = sz; i < hboxsz.size(); ++i) {
      hboxsz[i] = 0;
    }
  }
  // Determine the output shape. See if anything has to be done.
  casa::IPosition resShape(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    resShape[i] = shape[i] - hboxsz[i];
    if (resShape[i] <= 0) {
      if (!fillEdge) {
        return;
      }
      casa::Array<float> res1(shape);
      res1 = float();
      casa::Array<float> res2(shape);
      res2 = float();

      stat1 = std::move(res1);
      stat2 = std::move(res2);

      return;
    }
  }
  // Need to make shallow copy because operator() is non-const.
  casa::Array<float> arr(array);
  casa::Array<float> result1(resShape);
  casa::Array<float> result2(resShape);

  assert(result1.contiguousStorage());
  assert(result2.contiguousStorage());
  // Loop through all data and assemble as needed.

#pragma omp parallel for shared(result1, result2)
  for (int y = 0; y < shape[1] - hboxsz[1]; y++) {
    for (int x = 0; x < shape[0] - hboxsz[0]; x++) {
      casa::IPosition pos(ndim);
      casa::IPosition blc(ndim);
      casa::IPosition trc(ndim);

      pos[0] = x;
      pos[1] = y;
      // pos[2] = pos[3] = 0;

      blc[0] = pos[0];
      blc[1] = pos[1];
      // blc[2] = blc[3] = 0;

      trc[0] = blc[0] + hboxsz[0];
      trc[1] = blc[1] + hboxsz[1];
      // trc[2] = trc[3] = 0;

      float val1, val2;
      funcObj(arr(blc, trc), val1, val2);

      result1(pos) = val1;
      result2(pos) = val2;
#ifdef DEBUG_POS
      std::cerr << blc << trc << " pos: " << pos << std::endl;
#endif
    }
  }

  if (!fillEdge) {
    stat1 = std::move(result1);
    stat2 = std::move(result2);
    return;
  }
  casa::Array<float> fullResult1(shape);
  casa::Array<float> fullResult2(shape);
  fullResult1 = float();
  fullResult2 = float();
  hboxsz /= 2;
  fullResult1(hboxsz, resShape + hboxsz - 1) = result1;
  fullResult2(hboxsz, resShape + hboxsz - 1) = result2;

  stat1 = std::move(fullResult1);
  stat2 = std::move(fullResult2);
  // return fullResult;
}

template <typename DualStatFunc>
void onePassSlidingArrayMathFullBox(casa::Array<float> &stat1,
                             casa::Array<float> &stat2,
                             const casa::Array<float> &array,
                             const casa::IPosition &fullBoxSize,
                             const DualStatFunc &funcObj, bool fillEdge) {
  std::cout << fullBoxSize << std::endl;
  size_t ndim = array.ndim();
  const casa::IPosition &shape = array.shape();
  // Set full box size (-1) and resize/fill as needed.
  casa::IPosition boxsz(fullBoxSize);
  if (boxsz.size() != array.ndim()) {
    size_t sz = boxsz.size();
    boxsz.resize(array.ndim());
    for (size_t i = sz; i < boxsz.size(); ++i) {
      boxsz[i] = 0;
    }
  }
  // Determine the output shape. See if anything has to be done.
  casa::IPosition resShape(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    resShape[i] = shape[i] - boxsz[i];
    if (resShape[i] <= 0) {
      if (!fillEdge) {
        return;
      }
      casa::Array<float> res1(shape);
      res1 = float();
      casa::Array<float> res2(shape);
      res2 = float();

      stat1 = std::move(res1);
      stat2 = std::move(res2);

      return;
    }
  }
  // Need to make shallow copy because operator() is non-const.
  casa::Array<float> arr(array);
  casa::Array<float> result1(resShape);
  casa::Array<float> result2(resShape);

  assert(result1.contiguousStorage());
  assert(result2.contiguousStorage());
  // Loop through all data and assemble as needed.

#pragma omp parallel for shared(result1, result2)
  for (int y = 0; y < shape[1] - boxsz[1]; y++) {
    for (int x = 0; x < shape[0] - boxsz[0]; x++) {
      casa::IPosition pos(ndim);
      casa::IPosition blc(ndim);
      casa::IPosition trc(ndim);

      pos[0] = x;
      pos[1] = y;
      // pos[2] = pos[3] = 0;

      blc[0] = pos[0];
      blc[1] = pos[1];
      // blc[2] = blc[3] = 0;

      trc[0] = blc[0] + boxsz[0]-1;
      trc[1] = blc[1] + boxsz[1]-1;
      // trc[2] = trc[3] = 0;

      float val1, val2;
      funcObj(arr(blc, trc), val1, val2);

      result1(pos) = val1;
      result2(pos) = val2;
#ifdef DEBUG_POS
      std::cerr << blc << trc << " pos: " << pos << std::endl;
#endif
    }
  }

  if (!fillEdge) {
    stat1 = std::move(result1);
    stat2 = std::move(result2);
    return;
  }
  casa::Array<float> fullResult1(shape);
  casa::Array<float> fullResult2(shape);
  fullResult1 = float();
  fullResult2 = float();
  boxsz /= 2;
  fullResult1(boxsz, resShape + boxsz - 1) = result1;
  fullResult2(boxsz, resShape + boxsz - 1) = result2;

  stat1 = std::move(fullResult1);
  stat2 = std::move(fullResult2);
  // return fullResult;
}

void omp_onepass_median_madfm(casa::Array<casa::Float> &medians,
                              casa::Array<casa::Float> &madfm,
                              casa::Array<casa::Float> &matrix,
                              casa::IPosition &hboxsz) {
  std::cout << "One pass optimized call" << std::endl;

  onePassSlidingArrayMath(
      medians, madfm, matrix,
      hboxsz,
      StatFunc(), true);
}

void omp_onepass_median_madfm_fullbx(casa::Array<casa::Float> &medians,
                              casa::Array<casa::Float> &madfm,
                              casa::Array<casa::Float> &matrix,
                              casa::IPosition &boxsz) {
  std::cout << "One pass optimized call" << std::endl;

  onePassSlidingArrayMathFullBox(
      medians, madfm, matrix,
      boxsz,
      StatFunc(), true);
}
