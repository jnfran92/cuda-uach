


#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "../Utils/utils.h"
#include "../Utils/matutils.h"

#include "omp.h"

/*#define BSIZE 2*/




using namespace std;


typedef long int92;




 __global__ void btree_reduction(  double *gvector, int92 n_steps, double *ovector ){
	

	__shared__ double vector[BSIZE];

	int92 i = threadIdx.y;
	int92 j = threadIdx.x;

	int92 gi = threadIdx.y + blockDim.y*blockIdx.y;
	int92 gj = threadIdx.x + blockDim.x*blockIdx.x;

	vector[j] = gvector[gj];
	__syncthreads();
	
	int92 count = 2;
	
	for (int92 m=0; m < n_steps; m++){
		
		
		if( j % count == 0  ){
			vector[j] += vector[j + count/2  ];
		}

		count = 2*count; 
		__syncthreads();
	}


	gvector[gj] = vector[j];
	
	if (j == 0){
		ovector[blockIdx.x] = vector[j]; 
	}



}



__global__ void atomic_add(double *vector){

	__shared__ double res;

	res = 0.0;	
	int92 i = threadIdx.y;
	int92 j = threadIdx.x;

	int92 gi = threadIdx.y + blockDim.y*blockIdx.y;
	int92 gj = threadIdx.x + blockDim.x*blockIdx.x;

	/*double res;i*/
	/*res = 0;*/
	__syncthreads();


	atomicAdd(&res,vector[j]  );

	__syncthreads();
	vector[0] = res;


}


__global__ void atomic_reduction(double *vector, double *gres){

	__shared__ double res;

	res = 0.0;	
	/*int92 i = threadIdx.y;*/
	int92 j = threadIdx.x;

	/*int92 gi = threadIdx.y + blockDim.y*blockIdx.y;*/
	int92 gj = threadIdx.x + blockDim.x*blockIdx.x;

	__syncthreads();

	atomicAdd(&res,vector[j]  );

	__syncthreads();


	if(j==0){
	atomicAdd(gres, res);
	}

}



__global__ void btree_atomic_reduction(double *gvector, int92 n_steps ,double *gres){

	__shared__ double vector[BSIZE];
	
	/*int92 i = threadIdx.y;*/
	int92 j = threadIdx.x;

	/*int92 gi = threadIdx.y + blockDim.y*blockIdx.y;*/
	int92 gj = threadIdx.x + blockDim.x*blockIdx.x;

	vector[j] = gvector[gj];
	
	
	__syncthreads();
	
	
	int92 count = 2;
	

	for (int92 m=0; m < n_steps; m++){
		
		if( j % count == 0  ){
			vector[j] += vector[j + count/2  ];
		}

		count = 2*count; 
		__syncthreads();
	}



	__syncthreads();


	if(j==0){
	atomicAdd(gres, vector[0]);
	}

}


__global__ void shuffle_reduction(double *gvector, int92 n_steps ,double *gres){

	/*__shared__ double vector[BSIZE];*/
	
	/*int92 i = threadIdx.y;*/
	int92 j = threadIdx.x;

	/*int92 gi = threadIdx.y + blockDim.y*blockIdx.y;*/
	int92 gj = threadIdx.x + blockDim.x*blockIdx.x;

	/*vector[j] = gvector[gj];*/


	double thread_var;
	
	thread_var = gvector[gj];

	
	__syncthreads();
	
	
	int92 count = BSIZE/2;
	

	for (int92 m=0; m < n_steps; m++){
		
		/*if( j % count == 0  ){*/
			/*vector[j] += vector[j + count/2  ];*/
		/*}*/

		thread_var += __shfl_down_sync(0xffffffff, thread_var, count, BSIZE);

		count = count/2; 
		/*__syncthreads();*/
	}



	__syncthreads();


	if(j==0){
	atomicAdd(gres, thread_var);
	}

}





void reduce(double *vector, int92 n   ){



	int92 n_steps = (int92)log2((double) n) ;
	/*printf("n_steps: %ld \n", n_steps );*/

	int92 n_temp = n/2;

	for(int92 i=0; i<n_steps; i++ ){



		for (int92 j=0; j<n_temp ; j++){
			int92 idx = j*(n/n_temp);
			vector[idx] = vector[idx] + vector[idx + ( n/(n_temp*2) )  ];
		}

		n_temp = n_temp/2;

	}



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


	printf(" CUDA - Reduction \n");
	printf("BSIZE=%ld  \n",BSIZE );
	//Init parameters

	int92 n = atoi(argv[1]);

	int92 msize = sizeof(double)*n ;

	// Host Data

	double *a;
	double *b;
	/*double *c;*/

	printf("Request memory: %ld Mb   \n", msize/(1024*1024) );
	a = (double *) malloc( msize  );
	b = (double *) malloc( msize );

	// Create  Data
	for (int92 i=0; i<n; i++){

		a[i] = 1.0;
		b[i] = 0;
	}



	// CPU implementation--------------------------------


	float ms = 0.0;

	/*float t1=0.0, t2=0.0;*/

	/*print_dmatrix(a,1,n);*/


	cudaEventRecord(start);
	reduce(a, n);
	cudaEventRecord(stop);

	cudaEventElapsedTime(&ms, start, stop);
	printf("%d GPU - Result: %f Time: %lf  \n", ncpu,a[0], ms );


	/*print_dmatrix(a,1,n);*/




	// GPU Implementation--------------------------------


	// Create  Data
	for (int92 i=0; i<n; i++){

		a[i] = 1.0;
		b[i] = 0;
	}



	// Device Data

	double *a_dev;
	double *b_dev;

	HANDLE_ERROR( cudaMalloc((void **)&a_dev, msize)    );
	HANDLE_ERROR( cudaMalloc((void **)&b_dev, msize)    );
		
	// Copy Data to Device
	HANDLE_ERROR( cudaMemcpy(a_dev, a, msize  , cudaMemcpyHostToDevice   )  );


	ms = 0.0;
	int92 n_steps = (int92)log2((double)BSIZE);
	/*printf("%ld  \n",n_steps);*/


	int92 gsize = n;

	dim3 block(BSIZE,1,1);
	dim3 grid(n/BSIZE,1,1);

	cudaEventRecord(start);

	while(1){
		gsize= gsize/ BSIZE; 
		btree_reduction<<<gsize, block>>>(a_dev , n_steps, b_dev );
		cudaDeviceSynchronize();
	
		/*printf("gsize: %ld \n",gsize);     */
		
		if(gsize < BSIZE){
			
			if(gsize>1){
			
				
				cudaMemcpy(a_dev,b_dev, sizeof(double)*gsize , cudaMemcpyDeviceToDevice);
				atomic_add<<<1, gsize>>>(a_dev);
				cudaDeviceSynchronize();

			}
			
			break;
		}
		
		
		cudaMemcpy(a_dev,b_dev, sizeof(double)*gsize , cudaMemcpyDeviceToDevice);

	}
	
	cudaEventRecord(stop);


	// Retrieve Data from Device
	// Get data Devices
	HANDLE_ERROR(  cudaMemcpy(a, a_dev, msize, cudaMemcpyDeviceToHost )     );

	HANDLE_ERROR(  cudaMemcpy(b, b_dev, msize, cudaMemcpyDeviceToHost )     );
	/*print_dmatrix(a,1,n);*/
	/*printf("Reduction result: %f\n", a[0]);*/

	ms = 0;
	cudaEventElapsedTime(&ms, start, stop);
	printf("GPU Binary Tree %f, Time: %f \n",a[0] ,ms );
	



	// KERNEL 2

	// Create  Data
	for (int92 i=0; i<n; i++){

		a[i] = 1.0;
		/*b[i] = 0;*/
	}

	double *res_dev;	
	double res;

	res = 0.0;

	/*HANDLE_ERROR( cudaMalloc((void **)&a_dev, msize)    );*/
	HANDLE_ERROR( cudaMalloc((void **)&res_dev, sizeof(double))    );
		
	// Copy Data to Device
	HANDLE_ERROR( cudaMemcpy(a_dev, a, msize  , cudaMemcpyHostToDevice   )  );
	HANDLE_ERROR( cudaMemcpy(res_dev, &res, sizeof(double)  , cudaMemcpyHostToDevice   )  );
	
	ms = 0.0;

	dim3 block_atomic(BSIZE,1,1);
	dim3 grid_atomic(n/BSIZE,1,1);


	cudaEventRecord(start);
	atomic_reduction<<<grid_atomic, block_atomic >>>(a_dev, res_dev );
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	
	
	HANDLE_ERROR( cudaMemcpy(&res, res_dev, sizeof(double)  , cudaMemcpyDeviceToHost   )  );



	ms = 0;
	cudaEventElapsedTime(&ms, start, stop);
	printf("GPU Atomic Reduction %f, Time: %f \n",res ,ms );
	



	// KERNEL 3

	// Create  Data
	for (int92 i=0; i<n; i++){

		a[i] = 1.0;
		/*b[i] = 0;*/
	}

	/*double *res_dev;	*/
	/*double res;*/

	res = 0.0;

	/*HANDLE_ERROR( cudaMalloc((void **)&a_dev, msize)    );*/
	/*HANDLE_ERROR( cudaMalloc((void **)&res_dev, sizeof(double))    );*/
		
	// Copy Data to Device
	HANDLE_ERROR( cudaMemcpy(a_dev, a, msize  , cudaMemcpyHostToDevice   )  );
	HANDLE_ERROR( cudaMemcpy(res_dev, &res, sizeof(double)  , cudaMemcpyHostToDevice   )  );
	
	ms = 0.0;
	n_steps = (int92)log2((double)BSIZE);

	/*dim3 block_atomic(BSIZE,1,1);*/
	/*dim3 grid_atomic(n/BSIZE,1,1);*/


	cudaEventRecord(start);
	btree_atomic_reduction<<<grid_atomic, block_atomic >>>(a_dev,n_steps, res_dev );
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	
	
	HANDLE_ERROR( cudaMemcpy(&res, res_dev, sizeof(double)  , cudaMemcpyDeviceToHost   )  );



	ms = 0;
	cudaEventElapsedTime(&ms, start, stop);
	printf("GPU Btree Atomic Reduction %f, Time: %f \n",res ,ms );
	



	// KERNEL 4 SHUFFLE REDUCTION

	// Create  Data
	for (int92 i=0; i<n; i++){

		a[i] = 1.0;
		/*b[i] = 0;*/
	}

	/*double *res_dev;	*/
	/*double res;*/

	res = 0.0;

	/*HANDLE_ERROR( cudaMalloc((void **)&a_dev, msize)    );*/
	/*HANDLE_ERROR( cudaMalloc((void **)&res_dev, sizeof(double))    );*/
		
	// Copy Data to Device
	HANDLE_ERROR( cudaMemcpy(a_dev, a, msize  , cudaMemcpyHostToDevice   )  );
	HANDLE_ERROR( cudaMemcpy(res_dev, &res, sizeof(double)  , cudaMemcpyHostToDevice   )  );
	
	ms = 0.0;
	n_steps = (int92)log2((double)BSIZE);

	/*dim3 block_atomic(BSIZE,1,1);*/
	/*dim3 grid_atomic(n/BSIZE,1,1);*/


	cudaEventRecord(start);
	shuffle_reduction<<<grid_atomic, block_atomic >>>(a_dev,n_steps, res_dev );
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	
	
	HANDLE_ERROR( cudaMemcpy(&res, res_dev, sizeof(double)  , cudaMemcpyDeviceToHost   )  );



	ms = 0;
	cudaEventElapsedTime(&ms, start, stop);
	printf("GPU Shuffle Reduction %f, Time: %f \n",res ,ms );
	
	// Free memory

	cudaFree( a_dev );
	cudaFree( b_dev );

	free(a);
	free(b);





	return 0;
}





