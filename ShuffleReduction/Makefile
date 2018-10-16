
CC = nvcc
all:
		$(CC) -O3 -DBSIZE=${BSIZE}  -arch=sm_60 -Xcompiler -fopenmp reduction.cu -o prog
