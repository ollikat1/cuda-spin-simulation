# Makefile

# Variables
#CUDA_INSTALL_PATH ?= /appl/opt/cuda/6.0
NVCC ?= nvcc #$(CUDA_INSTALL_PATH)/bin/nvcc

PROGRAM  = spin2.gnu
LIBS     = -L"$(CUDA_INSTALL_PATH)/lib64" -lcufft -lcudart -lcublas #-lnvToolsExt
NVCCFLAGS := --restrict -arch=compute_20 -rdc=true -code=sm_20 #-lineinfo
#options for the c++ compiler (g++)
COMPILEROPTIONS = #-Xptxas -v -O3 #-std=c++11
OBJS     = spin2sim.cu

# Compile
$(PROGRAM): ${OBJS}
	$(NVCC) $(NVCCFLAGS) $(COMPILEROPTIONS) $(OBJS) $(LIBS) -o $(PROGRAM)
