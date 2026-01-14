.PHONY 01 01_obj_exe_run 02 03 clean

GCC = gcc 
NVCC = nvcc
CUDA_FLAGS = -arch=sm_86 

01: 
	@$(GCC) -o 01 01.c

01_obj:
	@$(GCC) -c 01.c -o 01.o 

01_obj_exe_run:
	@$(GCC) 01.o -o 01
	@./01

02:
	@$(GCC) -o 02 02.c

03: 
	@$(GCC) -o 03 03.c


