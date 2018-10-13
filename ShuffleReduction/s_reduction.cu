


#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "../Utils/utils.h"
#include "../Utils/matutils.h"

#include "omp.h"


using namespace std;


__global__ void julia_kernel( int n, double *a, double *b, double *c  ){

	int i = threadIdx.y;
	int j = threadIdx.x;

	int gi = threadIdx.y + blockDim.y*blockIdx.y;
	int gj = threadIdx.x + blockDim.x*blockIdx.x;




}



int main( int argc, char**  argv  ){

	int args_needed = 1;
	if (argc < args_needed + 1 ){
		printf(" Arg number error, needed: %d  \n", args_needed);
		return 0;
	}


	// Timers
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// OMP
	int ncpu = 1;
	omp_set_num_threads(ncpu);


	printf(" CUDA - Template \n");

	//Init parameters

	int n = 10;



	// Host Data

	double *a;
	double *b;
	double *c;


	a = (double *) malloc( sizeof(double)*n  );
	b = (double *) malloc( sizeof(double)*n );



	// Device Data

	double *a_dev;
	double *b_dev;
	double *c_dev;

	HANDLE_ERROR( cudaMalloc((void **)&a, sizeof(double)*n)    );



	// Copy Data to Device


	HANDLE_ERROR( cudaMemcpy(a_dev, a, sizeof(double) *n , cudaMemcpyHostToDevice   )  );


	// Kernel Implementation

	float ms = 0.0;

	dim3 block(1,1,1);
	dim3 grid(1,1,1);

	cudaEventRecord(start);
	julia_kernel<<<grid, block>>>(n, a_dev, b_dev, c_dev);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);


	// Retrieve Data from Device
	// Get data Devices
	HANDLE_ERROR(  cudaMemcpy(c, c_dev, sizeof(double) * n, cudaMemcpyDeviceToHost )     );




	ms = 0;
	cudaEventElapsedTime(&ms, start, stop);
	printf("GPU SM WHILE Time: %f\n", ms );


	// Free memory

	cudaFree( a_dev );
	cudaFree( b_dev );
	cudaFree( c_dev );

	free(a);
	free(b);
	free(c);





	return 0;
}





