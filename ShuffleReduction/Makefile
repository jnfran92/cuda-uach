#NCPU=1
CC = nvcc
all:
		$(CC) -O3 -DBSIZE=${BSIZE} -DNCPU=${NCPU}  -arch=sm_60 -Xcompiler -fopenmp reduction.cu -o prog
