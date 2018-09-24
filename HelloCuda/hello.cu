
#include <stdio.h>
#include <stdlib.h>
/*#include <iostream>*/
#include "utils.h"

__global__ void kernel( int a, int b, int *c ){
	printf("Hello from CUDA Core\n");
	*c = a + b;
}


int main(int argc, char **argv ) {

	printf("Hello CUDA \n");
	



	int count_cuda_devs;
	HANDLE_ERROR(cudaGetDeviceCount( &count_cuda_devs ));
	printf("Cuda-enabled devices =  %d  \n", count_cuda_devs);


	cudaDeviceProp prop;
	
	int i;
	for (i = 0; i < count_cuda_devs; i++){

		HANDLE_ERROR(cudaGetDeviceProperties( &prop, i ));
		printf("Device:%d\n	prop.name:  %s  \n",i,  prop.name);

		printf("	Memory: %zu  Mb \n", (prop.totalGlobalMem)/(1024*1024)   );
		
		printf( "	Clock rate: %d\n", prop.clockRate );

		printf("\n");

		printf("\n");

	}

	

	printf("Current device\n");
	int dev;
	HANDLE_ERROR( cudaGetDevice( &dev ) );
    	printf( "ID of current CUDA device:  %d\n", dev );
	

	int c;
	int *dev_c;

	cudaMalloc((void**)&dev_c, sizeof(int) );
	

	kernel<<<1,1>>>(999998,1,dev_c);


	cudaMemcpy(&c, dev_c, sizeof(int) , cudaMemcpyDeviceToHost  );

	printf("Res: %d  \n", c);

	cudaFree(dev_c);	

	return 0;
}








