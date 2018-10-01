


#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include "../Utils/utils.h"
#include "../Utils/matutils.h"

#include "omp.h"
#define PRINT 1


#define BSIZE 16

using namespace std;

//Pedir n como parametro
__global__ void matmul1( int n,  double *a, double *b, double *c){


	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	int idx = threadIdx.x + blockDim.x * blockIdx.x;


//	int n = blockDim.x * gridDim.x;
	int k;
	double r = 0.0;
	for ( k=0; k< n  ; k++   ){
	
				 
		r +=  a[ n*idy + k  ] *  b[ n*k + idx  ];

	}

	c[ n*idy + idx ] = r;

}

__global__ void matmul_sm(int n, double *a, double *b, double *c){

	__shared__ double as[BSIZE*BSIZE];
	__shared__ double bs[BSIZE*BSIZE];
	__shared__ double cs[BSIZE*BSIZE];

	int ltidx = threadIdx.x;
	int ltidy = threadIdx.y;
	
	int tidy = threadIdx.y + blockDim.y * blockIdx.y;
	int tidx = threadIdx.x + blockDim.x * blockIdx.x;

	
	cs[bx*ltidy + ltidx ] = 0;


	
	__syncthreads();

	




	as[bx*ltidy + ltidx ] = a[n*tidy  + tidx ];
	
	bs[bx*ltidy + ltidx ] = b[n*tidy  + tidx ];
	

	double r = 0.0;
	for ( k=0; k< BSIZE  ; k++   ){
		r +=  a[ n*idy + k  ] *  b[ n*k + idx  ];
	}

	c[ n*idy + idx ] = r;



	



	__syncthreads();
	




	


}

void matmulcpu(int n,  double *a, double *b, double *c){

//	int j, k;
	#pragma omp parallel for
	for ( int i = 0; i< n ;i++ ){
		for ( int j = 0; j< n ;j++ ){
			double temp = 0.0;			
			for (int k = 0; k< n ;k++ ){
	
				temp += a[i*n + k  ] * b[ i*n + k  ];

			}
			c[i*n +  j ] = temp;

		}	
	}

}



int main( int argc, char**  argv  ){

	int args_needed = 3;
	if (argc < args_needed + 1 ){
		printf(" Arg number error, needed: %d  \n", args_needed);
		return 0;	
	}


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	printf(" CUDA - Maxmul  \n");


	// Select Device
//	HANDLE_ERROR(  cudaSetDevice(0)  ) ;
	
	

	// Size
	int n = atoi(argv[1]);
	int nt = atoi(argv[2]);

	int ncpu = atoi(argv[3]);
	//Create Data host n x n

	double *a;
	double *b;
	double *c;	

	a = (double *)malloc( sizeof(double) * n * n  );
	b = (double *)malloc( sizeof(double) * n * n  );
	c = (double *)malloc( sizeof(double) * n * n  );

	int i,j;

	for ( i =0; i<n ; i++  ){
		for( j =0; j<n ; j++){
			a[ i*n + j  ] = i*n + j;
			b[ i*n + j ] = i*n + j;
			c[ i*n + j ] = 0;
		}
	}
	


	
//	print_dmatrix(a,n,n);
//	print_dmatrix(b,n,n);	


	omp_set_num_threads(ncpu);

	cudaEventRecord(start);	
	matmulcpu(n,a,b,c);
	cudaEventRecord(stop);


	
	float milliseconds1 = 0;
	cudaEventElapsedTime(&milliseconds1, start, stop);
	printf("Time CPU : %f\n", milliseconds1 );	






	// CUDA data
	double *a_dev;
	double *b_dev;
	double *c_dev;


	HANDLE_ERROR(  cudaMalloc((void **)&a_dev, sizeof(double) * n * n)   );

	HANDLE_ERROR(  cudaMalloc((void **)&b_dev, sizeof(double) * n * n)   );

	HANDLE_ERROR(  cudaMalloc((void **)&c_dev, sizeof(double) * n * n)   );


	// Memcpy
	
	HANDLE_ERROR(  cudaMemcpy(a_dev, a, sizeof(double) * n * n, cudaMemcpyHostToDevice )     );
	HANDLE_ERROR(  cudaMemcpy(b_dev, b, sizeof(double) * n * n, cudaMemcpyHostToDevice )     );	
	
	
	// Kernel
	
	dim3 block(nt, nt, 1);
	dim3 grid(n/nt, n/nt, 1);

	
	cudaEventRecord(start);
	matmul1<<<grid, block>>>(n,a_dev,b_dev,c_dev);
	cudaEventRecord(stop);

	// Get data Devices
	HANDLE_ERROR(  cudaMemcpy(c, c_dev, sizeof(double) * n * n, cudaMemcpyDeviceToHost )     );
	

	cudaEventSynchronize(stop);
	

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time: %f\n", milliseconds );	



	//Free
	cudaFree( a_dev );
	cudaFree( b_dev );
	cudaFree( c_dev );

	free(a);
	free(b);
	free(c);


	return 0;
}





//	printf("%d %d  \n", idy, idx);


	// int bid = (blockDim.x * blockDim.y) * ( gridblockIdx.y )

	//if(tid == 0){
	//	printf("bid %d   idx %d  \n ", bid, idx);
	//}
	//printf("tid %d/n", tid);

	//int tid = blockDim.y * threadIdx.y +   threadIdx.x;
	//int bid = gridDim.y * blockIdx.y + blockIdx.x; 

	//int idx = (blockDim.x * blockDim.y)*bid + tid;



//	 printf("%d %d - %f\n", idy, idx, r );
