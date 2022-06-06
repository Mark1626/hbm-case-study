#include <assert.h>
#include <casacore/casa/Arrays.h>
#include <casacore/casa/Arrays/IPosition.h>
#include <getopt.h>
#include <stdlib.h>

#include "bitonic_sliding_median.cuh"
#include "omp_onepass.cc"

// One to one assertion is not possible as window sizes are not equal between
// CASA and the current algorithm

#define TOLERANCE 0.01f

enum Option {
  TWO_N_PLUS_ONE,
  TWO_N_MINUS_ONE,
  EXCLUDE_CASA,
  EXCLUDE_OMP,
};

static int enable_2n_plus_1 = 0;
// static int enable_2n_minus_1 = 1;
static int disable_casa = 0;
static int disable_omp = 0;
static int enable_half_box = 0;
static int disable_full_box = 0;
static int disable_assertion = 0;
static int disable_gpu = 0;

static struct option long_option[] = {
    {"size", required_argument, 0, 's'},
    {"nocasa", no_argument, &disable_casa, 'c'},
    {"noomp", no_argument, &disable_omp, 'o'},
    {"plus1", no_argument, &enable_2n_plus_1, 'p'},
    // {"minus1", no_argument, &enable_2n_minus_1, 'm'},
    {"halfbx", no_argument, &enable_half_box, 'd'},
    {"nofullbx", no_argument, &disable_full_box, 'f'},
    {"noassert", no_argument, &disable_assertion, 'a'},
    {"nogpu", no_argument, &disable_gpu, 'g'},
    {"help", no_argument, 0, 'h'},
    {0, 0, 0, 0}};

void print_usage() {
  std::cout << "./tbitonic [OPTIONS]" << std::endl;
  std::cout << "\t--help: Print this message" << std::endl;
  std::cout << "\t--size: Size of the array" << std::endl;
  std::cout << "\t--nocasa: Disable CASA reference benchmark" << std::endl;
  std::cout << "\t--noomp: Disable OMP reference benchmark" << std::endl;
  std::cout << "\t--plus1: CASA window size is 2n+1 ie) 33" << std::endl;
  // std::cout << "\t--minus1: CASA window size is 2n-1 ie) 31" << std::endl;
  std::cout << "\t--halfbx: Enable half box stats" << std::endl;
  std::cout << "\t--nofullbx: Disable full box stats" << std::endl;
  std::cout << "\t--noassert: Disable assertions" << std::endl;
  std::cout << "\t--nogpu: Disable CPU" << std::endl;
}

class WallClock {
  std::chrono::steady_clock::time_point begin;

public:
  void tick() { begin = std::chrono::steady_clock::now(); }
  double elapsedTime() {
    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
        .count();
  }
};

void assertArray(casa::Array<casa::Float> &m1, casa::Array<casa::Float> &m2) {

  assert(m1.ndim() == m2.ndim());
  for (int i = 0; i < m1.ndim(); i++) {
    assert(m1.shape()[i] == m2.shape()[i]);
  }

  for (ssize_t i = 0; i < m1.shape()[0]; i++) {
    for (ssize_t j = 0; j < m1.shape()[1]; j++) {

      float val =
          fabs(m1(casa::IPosition(2, i, j)) - m2(casa::IPosition(2, i, j)));

      if (val > TOLERANCE) {
        std::cerr << "Assertion failed at " << std::endl;
        std::cout << "i : " << i << " j: " << j << " diff: " << val
                  << std::endl;
        std::cout << "expected: " << m1(casa::IPosition(2, i, j)) << " vs "
                  << "actual: " << m2(casa::IPosition(2, i, j)) << std::endl;
        exit(1);
      }
    }
  }
}

void reference_implementation(casacore::Array<float> &matrix,
                              casa::IPosition &hboxsz) {
  if (!disable_casa) {
    casacore::Array<float> medianCASA, madfmCASA;
    WallClock clock;
    clock.tick();

    medianCASA =
        casa::slidingArrayMath(matrix, hboxsz, casa::MedianFunc<casa::Float>());

    madfmCASA =
        casa::slidingArrayMath(matrix, hboxsz, casa::MadfmFunc<casa::Float>());

    std::cout << "CASA wall clock time taken: " << clock.elapsedTime() << "ms"
              << std::endl;
  }

  if (!disable_omp) {
    casacore::Array<float> medianOMP, madfmOMP;

    WallClock clock;
    clock.tick();

    omp_onepass_median_madfm(medianOMP, madfmOMP, matrix, hboxsz);

    std::cout << "OMP version wall clock time taken: " << clock.elapsedTime()
              << "ms" << std::endl;

    // std::cout << medianOMP << std::endl;
  }
}

void reference_implementation_fullbx(casacore::Array<float> &matrix,
                                     casa::IPosition &boxsz,
                                     casacore::Array<float> &medianOMP,
                                     casacore::Array<float> &madfmOMP) {
  WallClock clock;
  clock.tick();

  omp_onepass_median_madfm_fullbx(medianOMP, madfmOMP, matrix, boxsz);

  std::cout << "OMP version wall clock time taken: " << clock.elapsedTime()
            << "ms" << std::endl;
}

int main(int argc, char **argv) {
  size_t size = 64;

  if (argc < 2) {
    print_usage();
    return 1;
  }

  int option_index = 0;
  int c;
  while ((c = getopt_long(argc, argv, "s:copdah", long_option,
                          &option_index)) != -1) {
    option_index = 0;
    switch (c) {
    case 0:
      break;
    case 's':
      size = atoi(optarg);
      break;
    case 'h':
    case 'c':
    case 'o':
    case 'p':
    // case 'm':
    case 'd':
    case 'a':
    case 'f':
      break;
    default:
      printf("Defaulting %c %d\n", c, c);
      print_usage();
      return 1;
      break;
    }
  }

  casa::IPosition shape(2, size, size);

  casa::Array<casa::Float> matrix(shape);
  double a = 5.0;

  for (int j = 0; j < size; j++) {
    for (int i = 0; i < size; i++) {
      auto val = (double)std::rand() / (double)(RAND_MAX / a);
      matrix(casacore::IPosition(2, i, j)) = val;
      // matrix(casacore::IPosition(2, i, j)) = (j * size + i) * 1.0f;
    }
  }

#ifdef DEBUG
  std::cout << matrix << std::endl;
#endif

  if (enable_half_box) {
    if (enable_2n_plus_1) {
      std::cout << "2n+1" << std::endl;
      auto hboxsz =
          casa::IPosition(2, (WINDOW_SIZE + 1) / 2, (WINDOW_SIZE + 1) / 2);
      reference_implementation(matrix, hboxsz);
    } else {
      std::cout << "2n-1" << std::endl;
      auto hboxsz =
          casa::IPosition(2, (WINDOW_SIZE - 1) / 2, (WINDOW_SIZE - 1) / 2);
      reference_implementation(matrix, hboxsz);
    }
  }

  casa::Array<casa::Float> medianRef, madfmRef;
  if (!disable_full_box) {
    std::cout << "2n" << std::endl;
    WallClock clock;
    clock.tick();
    auto boxsz = casa::IPosition(2, WINDOW_SIZE, WINDOW_SIZE);

    reference_implementation_fullbx(matrix, boxsz, medianRef, madfmRef);

    std::cout << "Full box OMP time taken: " << clock.elapsedTime() << "ms"
              << std::endl;

    // std::cout << medianRef << std::endl;
  }

  casa::Array<casa::Float> medianCuda, madfmCuda;
  if (!disable_gpu) {
    std::cout << "2n" << std::endl;

    // Note Wall clock time is not accurate for GPUs internally the driver
    // will print the time taken based on Cuda Events, use that for measuring
    // performance

    WallClock clock;
    clock.tick();

    bitonicSlidingSNR(medianCuda, madfmCuda, matrix);

    std::cout << "GPU version wall clock time taken: " << clock.elapsedTime()
              << "ms" << std::endl;

    // std::cout << medianCuda << std::endl;
  }

  // One to one assertion is not possible as window sizes are not equal

  if (!disable_assertion) {
    std::cout << "Asserting Median" << std::endl;
    assertArray(medianRef, medianCuda);
    std::cout << "Asserting MADFM" << std::endl;
    assertArray(madfmRef, madfmCuda);
  }
}
