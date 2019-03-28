
CC = nvcc

matmul:
		$(CC) -O3  -DBSIZE=${BSIZE}   -Xcompiler -fopenmp matmul.cu -o prog
mems:
		$(CC) -O3  -DBSIZE=${BSIZE}   -Xcompiler -fopenmp mems.cu -o prog2


