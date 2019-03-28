
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "utils.h"

#include "omp.h"


using namespace std;

typedef float bebop;
typedef long int92;

__global__ void matmul_kernel( int92 n, bebop *gA, bebop *gB, bebop *gC  ){


	__shared__ bebop A[BSIZE*BSIZE];
	__shared__ bebop B[BSIZE*BSIZE];
	__shared__ bebop C[BSIZE*BSIZE];

	int92 i = threadIdx.y;
	int92 j = threadIdx.x;

	int92 gi = threadIdx.y + blockDim.y*blockIdx.y;
	int92 gj = threadIdx.x + blockDim.x*blockIdx.x;

	C[ BSIZE*i +j ] = 0.0;
	int92 K = n/BSIZE;
	int92 k;
	bebop sum;


	for(int92 step = 0; step<K ; step++){

		A[ BSIZE* i + j] = gA[n*gi + (j + step*BSIZE)  ];
		/*B[ BSIZE* i + j] = gB[n*gi + (j + step*BSIZE)  ]; */
		B[ BSIZE* i + j] = gB[n*(i + step*BSIZE) + gj  ];
		/*A[ BSIZE* i + j] = Cg[n*i + j  ]; */
		__syncthreads();


		sum = 0.0;
		for (k=0; k<BSIZE; k++ ){
			sum += A[BSIZE*i + k ] *  B[BSIZE*k + j  ];
			/*sum += A[BSIZE*i + k ] *  B[BSIZE*j + k  ];*/
			/*sum += A[BSIZE*k + j ] *  B[BSIZE*k + j  ];*/
		}



		/*__syncthreads();*/
		C[BSIZE*i + j] += sum;
		__syncthreads();
	}

	/*__syncthreads();\*/

	gC[n*gi + gj] = C[BSIZE*i + j];


}

void matmul_host(int92 n, bebop *A, bebop *B, bebop *C ){

	/*bebop sum;*/
	int92 i,j,k;
#pragma omp parallel for
	for (i=0; i<n ; i++){

		for (j=0; j<n; j++){

			bebop sum = 0.0;
			for (k=0; k<n; k++){
				sum += A[ n*i + k ]*B[ n*j + k];
			}

			C[i*n + j ] = sum;
		}

	}

}


void print_matrix(bebop *c ){

	/*for (int i=0; i<10; i++){*/
		/*printf("matrix = %f  \n", c[i] );*/
	/*}*/
}



int main( int argc, char**  argv  ){

	int args_needed = 2;
	if (argc < args_needed + 1 ){
		printf(" Arg number error, needed: %d \n[1]n size [2]ncpu  \n", args_needed);
		return 0;
	}


	printf("Matmul mGPU\n");


	// OMP
	int ncpu = atoi(argv[2]);
	omp_set_num_threads(ncpu);

	// OMP Timers
	double t1 = 0.0 , t2 = 0.0 ;

	//Init parameters
	int92 n = atoi(argv[1]);

	size_t sizef = sizeof(bebop);

	printf("n= %ld  BSIZE=  %d \n", n , BSIZE);
	int92 mem =  (n*n*4)/(1024*1024);
	printf("Memory required: %ld Mb  Total: %ld Mb  \n", mem, 3*mem );


	// --------------------- HOST -----------------------
	printf("----HOST----\n");
	// Host Data
	bebop *a;
	bebop *b;
	bebop *c;

	a = (bebop *) malloc( sizef*n*n  );
	b = (bebop *) malloc( sizef*n*n );
	c = (bebop *) malloc( sizef*n*n );

	// Init Data
	for (int92 i=0; i<n*n; i++){

		a[i] = (bebop) i;
		/*a[i] = (bebop) 1.0;*/
	}


	for (int92 i=0; i<n; i++){

		for (int92 j=0; j<n; j++){
			b[j*n + i ] = (bebop) i*n + j;
			/*b[j*n + i ] = 1.0;*/
		}
	}

	t1 = omp_get_wtime();
	matmul_host(n,a,b,c);
	t2 = omp_get_wtime();

	printf("CPU ncpu:%d  Time:%f sec\n",ncpu,t2-t1);

	print_matrix(c);

	// --------------------- END -----------------------




	// ------------- SINGLE GPU -----------------------
	printf("----SINGLE GPU----\n");


	for (int92 i=0; i<n; i++){
		for (int92 j=0; j<n; j++){
			b[n*i + j ] = (bebop) n*i + j;
			/*b[j*n + i ] = (bebop) i*n + j;*/
		}
	}


	// Timers
	cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));

	// Device Data
	bebop *a_dev;
	bebop *b_dev;
	bebop *c_dev;

	HANDLE_ERROR( cudaMalloc((void **)&a_dev, sizef*n*n)    );
	HANDLE_ERROR( cudaMalloc((void **)&b_dev, sizef*n*n)    );
	HANDLE_ERROR( cudaMalloc((void **)&c_dev, sizef*n*n)    );



	// Copy Data to Device


	HANDLE_ERROR( cudaMemcpy(a_dev, a, sizef *n*n , cudaMemcpyHostToDevice   )  );
	HANDLE_ERROR( cudaMemcpy(b_dev, b, sizef *n*n , cudaMemcpyHostToDevice   )  );

	// Kernel Implementation

	float ms = 0.0;

	dim3 block(BSIZE,BSIZE,1);
	dim3 grid(n/BSIZE,n/BSIZE,1);

	HANDLE_ERROR(cudaEventRecord(start, 0));

	matmul_kernel<<<grid, block>>>(n , a_dev, b_dev, c_dev);
	cudaDeviceSynchronize();

	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR( cudaEventSynchronize( stop ) );

	// Retrieve Data from Device
	// Get data Devices
	HANDLE_ERROR(  cudaMemcpy(c, c_dev, sizef * n*n, cudaMemcpyDeviceToHost )     );

	print_matrix(c);

	ms = 0.0;
	cudaEventElapsedTime(&ms, start, stop);
	printf("time: %f ms \n", ms );

	printf("speed up: %f \n", (t2-t1)*1000.0/(ms) );



	HANDLE_ERROR( cudaEventDestroy( start ) );
	HANDLE_ERROR( cudaEventDestroy( stop ) );

	cudaFree( a_dev );
	cudaFree( b_dev );
	cudaFree( c_dev );

	// --------------------- END -----------------------






	// ------------- Naive MULTI GPU -----------------------
	printf("----naive MULTI GPU----\n");

	int deviceCount = 0;
	HANDLE_ERROR( cudaGetDeviceCount( &deviceCount ) );
	printf("CUDA devices:  %d \n", deviceCount);

	// Timers
	/*cudaEvent_t start, stop;*/
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));

	// Device Data
	bebop *a_mdev[2];
	bebop *b_mdev[2];
	bebop *c_mdev[2];


	for (int i=0; i<deviceCount; i++){

		HANDLE_ERROR( cudaSetDevice( i  ));

		HANDLE_ERROR( cudaMalloc((void **)&a_mdev[i], sizef*n*n)    );
		HANDLE_ERROR( cudaMalloc((void **)&b_mdev[i], sizef*n*n)    );
		HANDLE_ERROR( cudaMalloc((void **)&c_mdev[i], sizef*n*n)    );
	}


	// Copy Data to Device


	for (int i=0; i<deviceCount; i++){

		HANDLE_ERROR( cudaSetDevice( i  ));

		HANDLE_ERROR( cudaMemcpy(a_mdev[i], a, sizef *n*n , cudaMemcpyHostToDevice   )  );
		HANDLE_ERROR( cudaMemcpy(b_mdev[i], b, sizef *n*n , cudaMemcpyHostToDevice   )  );
	}
	// Kernel Implementation

	ms = 0.0;


	for (int i=0; i<deviceCount; i++){

		HANDLE_ERROR( cudaSetDevice( i  ));

		/*HANDLE_ERROR(cudaEventRecord(start, 0));*/

		matmul_kernel<<<grid, block>>>(n , a_mdev[i], b_mdev[i], c_mdev[i]);

	}
	cudaDeviceSynchronize();

	/*HANDLE_ERROR(cudaEventRecord(stop, 0));*/
	/*HANDLE_ERROR( cudaEventSynchronize( stop ) );*/

	// Retrieve Data from Device
	// Get data Devices


	for (int i=0; i<deviceCount; i++){

		HANDLE_ERROR( cudaSetDevice( i  ));

		HANDLE_ERROR(  cudaMemcpy(c, c_mdev[i], sizef * n*n, cudaMemcpyDeviceToHost )     );
		printf("\n");
		print_matrix(c);

	}


	ms = 0.0;
	cudaEventElapsedTime(&ms, start, stop);
	printf("time: %f ms \n", ms );

	printf("speed up: %f \n", (t2-t1)*1000.0/(ms) );



	HANDLE_ERROR( cudaEventDestroy( start ) );
	HANDLE_ERROR( cudaEventDestroy( stop ) );


	for (int i=0; i<deviceCount; i++){

		HANDLE_ERROR( cudaSetDevice( i  ));
		cudaFree( a_mdev[i] );
		cudaFree( b_mdev[i] );
		cudaFree( c_mdev[i] );
	}


	free(a);
	free(b);
	free(c);

	// --------------------- END -----------------------





	// ----------------- Single GPU UM -----------------
	printf("---SINGLE GPU UM--- \n");


	HANDLE_ERROR( cudaSetDevice( 0  ));
	// Timers
	/*cudaEvent_t start2, stop2;*/
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Unified Memory
	bebop *A, *B, *C;

	cudaMallocManaged( (void **)&A, sizef*n*n  );
	cudaMallocManaged( (void **)&B, sizef*n*n  );
	cudaMallocManaged( (void **)&C, sizef*n*n  );

	for (int92 i = 0; i < n*n; i++) {
		A[i] = (bebop) i;
		B[i] = (bebop) i;
		C[i] = 0;
	}


	// kernel

	/*dim3 block(BSIZE,BSIZE,1);*/
	/*dim3 grid(n/BSIZE,n/BSIZE,1);*/

	HANDLE_ERROR(	cudaEventRecord(start, 0) );

	matmul_kernel<<<grid, block>>>(n , A, B, C);
	cudaDeviceSynchronize();

	HANDLE_ERROR( cudaEventRecord(stop, 0) );
	HANDLE_ERROR( cudaEventSynchronize( stop ) );

	// Print Data
	print_matrix(C);


	ms = 0.0;
	cudaEventElapsedTime(&ms, start, stop);
	printf("time: %f ms \n", ms );

	printf("speed up: %f \n", (t2-t1)*1000.0/(ms) );

	HANDLE_ERROR( cudaEventDestroy( start ) );
	HANDLE_ERROR( cudaEventDestroy( stop ) );

	cudaFree(A);
	cudaFree(B);
	cudaFree(C);

	// ----------------- END -------------------------




	// ----------------- Multiple GPU UM -----------------
	printf("---MULTIPLE GPU UM--- \n");


	// Get number of devices supporting CUDA
	deviceCount = 0;
	HANDLE_ERROR( cudaGetDeviceCount( &deviceCount ) );
	printf("CUDA devices:  %d \n", deviceCount);



	// Timers
	/*cudaEvent_t start2, stop2;*/
	/*cudaEventCreate(&start);*/
	/*cudaEventCreate(&stop);*/

	// Unified Memory
	bebop *mA[2], *mB[2], *mC[2];

	/*bebop *mC[2];*/



	for (int i = 0; i<deviceCount; i++){

		cudaMallocManaged( (void **)&mA[i], sizef*n*n  );
		cudaMallocManaged( (void **)&mB[i], sizef*n*n  );
		cudaMallocManaged( (void **)&mC[i], sizef*n*n  );


		for (int92 j = 0; j < n*n; j++) {
			mA[i][j] = (bebop) j;
			mB[i][j] = (bebop) j;
			mC[i][j] = 0;
		}
	}

	// kernel

	/*dim3 block(BSIZE,BSIZE,1);*/
	/*dim3 grid(n/BSIZE,n/BSIZE,1);*/

	/*HANDLE_ERROR(	cudaEventRecord(start, 0) );*/

	/*matmul_kernel<<<grid, block>>>(n , A, B, C);*/
	/*cudaDeviceSynchronize();*/



	for (int i = 0; i<deviceCount; i++){
		printf("Kernel  \n");
		HANDLE_ERROR( cudaSetDevice( i  ));

		/*dim3 block(BSIZE,BSIZE,1);*/
		/*dim3 grid((n/2)/BSIZE,(n/2)/BSIZE,1);*/

		matmul_kernel<<<grid, block>>>(n , mA[i], mB[i], mC[i]);

	}

	cudaDeviceSynchronize();


	/*HANDLE_ERROR( cudaEventRecord(stop, 0) );*/
	/*HANDLE_ERROR( cudaEventSynchronize( stop ) );*/

	// Print Data
	for (int i = 0; i<deviceCount; i++){
		printf("  \n");
		print_matrix(mC[i]);
	}


	/*ms = 0.0;*/
	/*cudaEventElapsedTime(&ms, start, stop);*/
	/*printf("time: %f ms \n", ms );*/

	/*printf("speed up: %f \n", (t2-t1)*1000.0/(ms) );*/

	/*HANDLE_ERROR( cudaEventDestroy( start ) );*/
	/*HANDLE_ERROR( cudaEventDestroy( stop ) );*/

	for (int i = 0; i<deviceCount; i++){
		cudaFree(mA[i]);
		cudaFree(mB[i]);
		cudaFree(mC[i]);
	}
	// ----------------- END -------------------------








	return 0;




}





