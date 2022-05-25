HIPIFY=hipify-perl
HIPCC=hipcc
NVCC=nvcc
FLAGS=-O3

# all: all-cuda all-hip

all-cuda: 0_Vendor/coalesce-cuda \
					0_Vendor/transpose-cuda

all-hip: 0_Vendor/coalesce-hip \
					0_Vendor/transpose-hip


%.cpp: %.cu
	$(HIPIFY) $< > $@

%-hip: %.cpp
	$(HIPCC) -o $@ $< $(FLAGS)

%-cuda: %.cu
	$(NVCC) -o $@ $< $(FLAGS)

clean:
	@rm -f 0_Vendor/coalesce-cuda \
					0_Vendor/transpose-cuda \
					0_Vendor/coalesce-hip \
					0_Vendor/transpose-hip

.PHONY: all-cuda all-hip
