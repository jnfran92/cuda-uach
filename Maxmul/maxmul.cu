


#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include "../Utils/utils.h"
#include "../Utils/matutils.h"

#define PRINT 1

using namespace std;


typedef struct {
		int m;
		int n;
		double *elements;
		} Matrix;

__global__ void matmul1(double *a, double *b, double *c){


	int idy = threadIdx.y + blockDim.y * blockIdx.y;
	int idx = threadIdx.x + blockDim.x * blockIdx.x;


	int n = blockDim.x * gridDim.x;
	int k;
	double r = 0;
	for ( k=0; k< n  ; k++   ){
	
				 
		r +=  a[ n * idy + k  ] *  b[ n*k + idx  ];

	}

	c[ n * idy + idx ] = r;

}



__global__ void matmul2(Matrix A, Matrix B, Matrix C ){



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


	// Select Device
//	HANDLE_ERROR(  cudaSetDevice(0)  ) ;
	
	

	// Size
	int n = atoi(argv[1]);
	int nt = atoi(argv[2]);

	//Create Data host n x n

	double *a;
	double *b;
	double *c;	

	a = (double *)malloc( sizeof(double) * n * n  );
	b = (double *)malloc( sizeof(double) * n * n  );
	c = (double *)malloc( sizeof(double) * n * n  );

	int i;

	for ( i =0; i<n*n ; i++  ){
		a[i] = i;
		b[i] = i;
		c[i] = 0;
	}

	
//	print_dmatrix(a,n,n);
//	print_dmatrix(b,n,n);	

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
	
	dim3 threads(nt, nt, 1);
	dim3 blocks(n/nt, n/nt, 1);

	
	cudaEventRecord(start);
	matmul1<<< blocks, threads >>>(a_dev, b_dev, c_dev);
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


//	print_dmatrix(c,n,n);


//	Matrix A;
//	Matrix B;
//	Matrix C;


//	A.m = n;
//	A.n = n;
//	A.elements = (double *)malloc( sizeof(double) * n * n   );   
//	A.elements = a;
	
//	B.m = n;
// 	B.n = n;
// 	B.elements = (double *)malloc(  sizeof(double) * n * n );
// 	B.elements = b;

		
// 	C.m = n;
// 	C.n = n;
// 	C.elements = (double *)malloc(  sizeof(double) * n * n );
// 	C.elements = c;
	


// 	Matrix A_dev;
// 	Matrix B_dev;
// 	Matrix C_dev;

	
// 	A_dev.elements = (double *)malloc( sizeof(double) * n * n   );   
	
// 	B_dev.elements = (double *)malloc(  sizeof(double) * n * n );
	
// 	C_dev.elements = (double *)malloc(  sizeof(double) * n * n );
	




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
