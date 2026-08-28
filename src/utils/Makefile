CU_FILES :=$(shell find ./ -name "*.cu")
EXE_FILES:= $(CU_FILES:.cu=)

all:$(EXE_FILES)

# These build outside CMake, so HPC_TARGET_ARCH has to be given here: utils.cuh
# includes utils.h, which refuses to compile without it. It matches -arch below.
$(EXE_FILES):%:%.cu $(CU_FILES)
	nvcc -o $@ $< -O2 -std=c++17 --expt-relaxed-constexpr -lineinfo -arch=sm_90a -DHPC_TARGET_ARCH=90 -I../../ -I../../3rd/cutlass/include -lcuda
