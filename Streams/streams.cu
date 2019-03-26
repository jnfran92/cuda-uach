


#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "../Utils/utils.h"
#include "../Utils/matutils.h"

#include "omp.h"

const int N = 1 << 20;


using namespace std;



__global__ void kernel(float *x, int n)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
		x[i] = sqrt(pow(3.14159,i));
	}
}



int main()
{


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	const int num_streams = 8;

	cudaStream_t streams[num_streams];
	float *data[num_streams];



	cudaEventRecord(start);

	for (int i = 0; i < num_streams; i++) {
		cudaStreamCreate(&streams[i]);

		cudaMalloc(&data[i], N * sizeof(float));

		// launch one worker kernel per stream
		kernel<<<1, 64, 0, streams[i]>>>(data[i], N);

		// launch a dummy kernel on the default stream
		kernel<<<1, 1>>>(0, 0);
	}


	cudaDeviceSynchronize();


	cudaEventRecord(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time elapsed: %f   \n", milliseconds);




	cudaDeviceReset();

	return 0;
}
