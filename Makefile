HIPIFY=hipify-perl
HIPCC=hipcc
NVCC=nvcc
FLAGS=-O3

# all: all-cuda all-hip

all-cuda: 0_Vendor/coalesce-cuda \
					0_Vendor/transpose-cuda \
					1_Data_Trn/copy-cuda \
					3_Algo/bitonic-cuda

all-hip: 0_Vendor/coalesce-hip \
					0_Vendor/transpose-hip \
					1_Data_Trn/copy-hip \
					3_Algo/bitonic-hip


%.cpp: %.cu
	$(HIPIFY) $< > $@

%-hip: %.cpp
	$(HIPCC) -o $@ $< $(FLAGS)

%-cuda: %.cu
	$(NVCC) -o $@ $< $(FLAGS)

clean:
	@rm -f 0_Vendor/coalesce-cuda \
					0_Vendor/transpose-cuda \
					1_Data_Trn/copy-cuda \
					3_Algo/bitonic-cuda \
					0_Vendor/coalesce-hip \
					0_Vendor/transpose-hip \
					1_Data_Trn/copy-hip \
					3_Algo/bitonic-hip


.PHONY: all-cuda all-hip
