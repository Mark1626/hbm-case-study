cmake_minimum_required(VERSION 3.12)

project(BitonicSlidingMedian LANGUAGES CXX)

# include(CMakeFindDependencyMacro)
# find_dependency(Casacore REQUIRED COMPONENTS casa)

# find_package(casacore REQUIRED COMPONENTS casa)

set(CMAKE_BUILD_TYPE Release)

OPTION (USE_CUDA "Use Cuda" ON)
OPTION (USE_HIP "Use HIP" OFF)

OPTION (USE_OpenMP "Use OpenMP" ON)
IF(USE_OpenMP)
  FIND_PACKAGE(OpenMP)
  IF(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  ENDIF()
ENDIF()

include(CTest)

option(BUILD_SHARED_LIBS "Build as a shared library" OFF)

IF(USE_CUDA)

project(BitonicSlidingMedian LANGUAGES CXX CUDA)

add_library(bitonicSliding STATIC
  bitonic_sliding_median.cuh
  bitonic_sliding_median.cu
  bitonic_driver.cc
  )
target_compile_features(bitonicSliding PUBLIC cxx_std_11)
set_target_properties(bitonicSliding
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
ENDIF()

IF(USE_HIP)

if (NOT DEFINED ROCM_PATH )
     set ( ROCM_PATH "/opt/rocm"  CACHE STRING "Default ROCM installation directory." )
endif ()

list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)

find_package(HIP REQUIRED CONFIG PATHS ${ROCM_PATH} /opt/rocm)

set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})

add_library(bitonicSliding STATIC
  bitonic_sliding_median.cuh
  bitonic_sliding_median.cc
  bitonic_driver.cc
  )

target_compile_options(bitonicSliding PRIVATE -fgpu-rdc)
target_link_libraries(bitonicSliding PRIVATE -fgpu-rdc)
target_include_directories(bitonicSliding PRIVATE ${ROCM_PATH}/hsa/include)

ENDIF()



if(BUILD_TESTING)
  add_executable(tbitonic tbitonic.cc)

  target_link_libraries(tbitonic PRIVATE bitonicSliding -lcasa_casa OpenMP::OpenMP_CXX)

  IF(USE_CUDA)
    set_target_properties(tbitonic PROPERTIES CUDA_SEPERABLE_COMPILATION ON)
  ENDIF()

  IF(USE_HIP)
    target_compile_options(tbitonic PRIVATE -fgpu-rdc)
    target_link_libraries(tbitonic PRIVATE bitonicSliding -lcasa_casa OpenMP::OpenMP_CXX)
    target_link_libraries(tbitonic PRIVATE -fgpu-rdc hip::host)
  ENDIF()

  add_test(NAME assert COMMAND tbitonic)

endif()
