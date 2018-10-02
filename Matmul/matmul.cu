


#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include "../Utils/utils.h"
#include "../Utils/matutils.h"

#include "omp.h"
#define PRINT 1


//#define BSIZE 2

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

	int i = threadIdx.y;
	int j = threadIdx.x;
	
	int gi = threadIdx.y + blockDim.y*blockIdx.y;
	int gj = threadIdx.x + blockDim.x*blockIdx.x;
 
	cs[BSIZE*i + j] = 0;

	__syncthreads();

	int l;
	for(l=0; l<n/BSIZE; l++){
		// Write in cache memory 
		int offset = l*BSIZE;	
		as[BSIZE*i + j] = a[n*gi + (offset+j)];
		bs[BSIZE*i + j] = b[n*(offset+i) + gj];

		__syncthreads();
		// Block matmul
		double r = 0.0;
		int k;
		for ( k=0; k< BSIZE  ; k++   ){
			r +=  as[ BSIZE*i + k  ] *  bs[ BSIZE*k + j ];
		}
		cs[ BSIZE*i +j ] += r;
		//End work of 1`st stage
	__syncthreads();
	}

	//Write Global from cache
	c[n*gi + gj] = cs[BSIZE*i + j];


}

void matmulcpu(int n,  double *a, double *b, double *c){

//	int j, k;
	#pragma omp parallel for
	for ( int i = 0; i< n ;i++ ){
		for ( int j = 0; j< n ;j++ ){
			double temp = 0.0;			
			for (int k = 0; k< n ;k++ ){
	
				temp += a[n*i + k  ] * b[ n*k + j  ];

			}
			c[i*n +  j ] = temp;

		}	
	}

}


void matmulcpu_transp(int n,  double *a, double *b, double *c){

//	int j, k;
	#pragma omp parallel for
	for ( int i = 0; i< n ;i++ ){
		for ( int j = 0; j< n ;j++ ){
			double temp = 0.0;			
			for (int k = 0; k< n ;k++ ){
	
				temp += a[n*i + k  ] * b[ n*i + k  ];

			}
			c[i*n +  j ] = temp;

		}	
	}

}





int main( int argc, char**  argv  ){

	int args_needed = 2;
	if (argc < args_needed + 1 ){
		printf(" Arg number error, needed: %d  \n", args_needed);
		return 0;	
	}


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	printf(" CUDA - Maxmul  \n");

	// Size
	int n = atoi(argv[1]);
	int ncpu = atoi(argv[2]);


	printf("BSIZE:  %d - N %d - #CPU %d \n", BSIZE, n, ncpu);

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

//	print_dmatrix(c,n,n);
	
	float milliseconds1 = 0;
	cudaEventElapsedTime(&milliseconds1, start, stop);
	printf("Time CPU : %f\n", milliseconds1 );	


	// Transpose CPU

	cudaEventRecord(start);	
	matmulcpu_transp(n,a,b,c);
	cudaEventRecord(stop);

//	print_dmatrix(c,n,n);
	
	milliseconds1 = 0;
	cudaEventElapsedTime(&milliseconds1, start, stop);
	printf("Time CPU Transp : %f\n", milliseconds1 );	






	//printf("*\n");


	//Rset C

	for ( i =0; i<n ; i++  ){
		for( j =0; j<n ; j++){
			c[ i*n + j ] = 0;
		}
	}
	

	// CUDA data
	double *a_dev;
	double *b_dev;
	double *c_dev;


	HANDLE_ERROR(  cudaMalloc((void **)&a_dev, sizeof(double) * n * n)   );

	HANDLE_ERROR(  cudaMalloc((void **)&b_dev, sizeof(double) * n * n)   );

	HANDLE_ERROR(  cudaMalloc((void **)&c_dev, sizeof(double) * n * n)   );


	//printf("*\n");
	// Memcpy
	
	HANDLE_ERROR(  cudaMemcpy(a_dev, a, sizeof(double) * n * n, cudaMemcpyHostToDevice )     );
	HANDLE_ERROR(  cudaMemcpy(b_dev, b, sizeof(double) * n * n, cudaMemcpyHostToDevice )     );	
	
	
	//printf("*\n");
	// Kernel
	
	dim3 block(BSIZE, BSIZE, 1);
	dim3 grid(n/BSIZE, n/BSIZE, 1);

	
	//printf("*\n");
	cudaEventRecord(start);
	matmul1<<<grid, block>>>(n,a_dev,b_dev,c_dev);
	cudaEventRecord(stop);


	cudaEventSynchronize(stop);
	
//	printf("*\n");
	// Get data Devices
	HANDLE_ERROR(  cudaMemcpy(c, c_dev, sizeof(double) * n * n, cudaMemcpyDeviceToHost )     );
	

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("GPU NoSM  Time: %f\n", milliseconds );	



	// MATMUL KERNEL 1

	//printf("*\n");
	cudaEventRecord(start);
	matmul_sm<<<grid, block>>>(n,a_dev,b_dev,c_dev);
	cudaEventRecord(stop);


	cudaEventSynchronize(stop);
	
//	printf("*\n");
	// Get data Devices
	HANDLE_ERROR(  cudaMemcpy(c, c_dev, sizeof(double) * n * n, cudaMemcpyDeviceToHost )     );
	

	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("GPU SM Time: %f\n", milliseconds );	





//	print_dmatrix(c,n,n);

	//Free
	cudaFree( a_dev );
	cudaFree( b_dev );
	cudaFree( c_dev );

	free(a);
	free(b);
	free(c);


	return 0;
}


