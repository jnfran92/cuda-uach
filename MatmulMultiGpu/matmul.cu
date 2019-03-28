
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "utils.h"

#include "omp.h"

#define PRINT 0


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
		B[ BSIZE* i + j] = gB[n*(i + step*BSIZE) + gj  ];
		__syncthreads();


		sum = 0.0;
		for (k=0; k<BSIZE; k++ ){
			sum += A[BSIZE*i + k ] *  B[BSIZE*k + j  ];
		}



		C[BSIZE*i + j] += sum;
		__syncthreads();
	}


	gC[n*gi + gj] = C[BSIZE*i + j];


}






void matmul_host(int92 n, bebop *A, bebop *B, bebop *C ){

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
	if(PRINT==1){
		for (int i=0; i<10; i++){
			printf("matrix = %f  \n", c[i] );
		}
	}
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
	printf("Memory required: %ld Mb  Total(A+B+C): %ld Mb  \n", mem, 3*mem );



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
	int92 p, q;
#pragma omp parallel for
	for (p=0; p<n*n; p++){

		a[p] = (bebop) p;
		/*a[i] = (bebop) 1.0;*/
	}


#pragma omp parallel for
	for (p=0; p<n; p++){

		for (q=0; q<n; q++){
			b[q*n + p ] = (bebop) p*n + q;
			/*b[j*n + i ] = 1.0;*/
		}
	}

	t1 = omp_get_wtime();
	/*matmul_host(n,a,b,c);*/
	t2 = omp_get_wtime();

	print_matrix(c);
	printf("CPU ncpu:%d  Time:%f sec\n",ncpu,t2-t1);






	if(mem <= 15000){


	// --------- GPU -----------------
	printf("------- only-GPU ----\n");

	// Regular B, not B Transp
	for (int92 i=0; i<n; i++){
		for (int92 j=0; j<n; j++){
			b[n*i + j ] = (bebop) n*i + j;
		}
	}




	// Device Data
	bebop *a_dev;
	bebop *b_dev;
	bebop *c_dev;

	// Timers
	cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	HANDLE_ERROR( cudaMalloc((void **)&a_dev, sizef*n*n)    );
	HANDLE_ERROR( cudaMalloc((void **)&b_dev, sizef*n*n)    );
	HANDLE_ERROR( cudaMalloc((void **)&c_dev, sizef*n*n)    );

	// Copy Data HtD
	HANDLE_ERROR( cudaMemcpy(a_dev, a, sizef *n*n , cudaMemcpyHostToDevice   )  );
	HANDLE_ERROR( cudaMemcpy(b_dev, b, sizef *n*n , cudaMemcpyHostToDevice   )  );


	// Kernel Implementation

	float ms = 0.0;

	dim3 block(BSIZE,BSIZE,1);
	dim3 grid(n/BSIZE,n/BSIZE,1);


	matmul_kernel<<<grid, block>>>(n , a_dev, b_dev, c_dev);
	cudaDeviceSynchronize();

	// Retrieve Data from Device
	// Get data Devices
	HANDLE_ERROR(  cudaMemcpy(c, c_dev, sizef * n*n, cudaMemcpyDeviceToHost )     );



	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR( cudaEventSynchronize( stop ) );


	print_matrix(c);


	ms = 0.0;
	cudaEventElapsedTime(&ms, start, stop);
	printf("GPU Time: %f ms \n", ms );

	printf("speed up: %f \n", (t2-t1)*1000.0/(ms) );



	HANDLE_ERROR( cudaEventDestroy( start ) );
	HANDLE_ERROR( cudaEventDestroy( stop ) );

	cudaFree( a_dev );
	cudaFree( b_dev );
	cudaFree( c_dev );



	}







	// --------- mGPU -----------------
	printf("------- CPU + mGPU ----\n");

	//C with zeros
	for (int92 i=0; i<n*n; i++){
		c[i] = (bebop) 0;
	}

	// Number of Devices
	int deviceCount = 0;
	HANDLE_ERROR( cudaGetDeviceCount( &deviceCount ) );
	printf("CUDA devices:  %d \n", deviceCount);

	//Host Aux Matrices
	bebop *a_aux;
	bebop *b_aux;
	bebop *c_aux;

	//Request memory
	/*a_aux = (bebop *) malloc(sizef*n*n/4);*/
	/*b_aux = (bebop *) malloc(sizef*n*n/4);*/
	/*c_aux = (bebop *) malloc(sizef*n*n/4);*/

	HANDLE_ERROR( cudaHostAlloc( (void**)&a_aux, sizef *n*n/4 , cudaHostAllocDefault ) );
	HANDLE_ERROR( cudaHostAlloc( (void**)&b_aux, sizef *n*n/4 , cudaHostAllocDefault ) );
	HANDLE_ERROR( cudaHostAlloc( (void**)&c_aux, sizef *n*n/4 , cudaHostAllocDefault ) );



	// Device Data
	bebop *a_mdev[2];
	bebop *b_mdev[2];
	bebop *c_mdev[2];

	cudaEvent_t mstart[2], mstop[2];

	for (int i=0; i<deviceCount; i++){

		HANDLE_ERROR( cudaSetDevice( i  ));

		HANDLE_ERROR( cudaEventCreate(&mstart[i]) );
		HANDLE_ERROR( cudaEventCreate(&mstop[i]) );
		HANDLE_ERROR( cudaEventRecord(mstart[i], 0) );
		
		
		HANDLE_ERROR( cudaMalloc((void **)&a_mdev[i], sizef*n*n/4)    );
		HANDLE_ERROR( cudaMalloc((void **)&b_mdev[i], sizef*n*n/4)    );
		HANDLE_ERROR( cudaMalloc((void **)&c_mdev[i], sizef*n*n/4)    );

	}


	// Start
	/*int92 k =0;*/

	for (int gi=0 ; gi<2 ; gi++){
		for (int gj=0 ; gj<2 ; gj++){

			for (int k=0; k<deviceCount; k++){

#pragma omp parallel for
				//Get Data from a and b
				for (p=0; p<n/2; p++){
					for (q=0; q<n/2; q++){
						a_aux[(n/2)*p + q  ] = a[n*(p + gi*n/2) + (q+ k*n/2 )  ];
						b_aux[(n/2)*p + q  ] = b[n*(p + k*n/2) + (q + gj*n/2)  ];
						c_aux[(n/2)*p + q  ] = 0.0;
					}
				}


				HANDLE_ERROR( cudaSetDevice( k  ));

				HANDLE_ERROR( cudaMemcpy(a_mdev[k], a_aux, sizef *n*n/4 , cudaMemcpyHostToDevice   )  );
				HANDLE_ERROR( cudaMemcpy(b_mdev[k], b_aux, sizef *n*n/4 , cudaMemcpyHostToDevice   )  );
			}


			dim3 mblock(BSIZE,BSIZE,1);
			dim3 mgrid((n/2)/BSIZE,(n/2)/BSIZE,1);


			for (int i=0; i<deviceCount; i++){

				HANDLE_ERROR( cudaSetDevice( i  ));
			
				
				matmul_kernel<<<mgrid, mblock>>>(n/2 , a_mdev[i], b_mdev[i], c_mdev[i]);
			
				
			
			}
			cudaDeviceSynchronize();


			for (int k=0; k<deviceCount; k++){

				HANDLE_ERROR( cudaSetDevice( k  ));

				HANDLE_ERROR(  cudaMemcpy(c_aux, c_mdev[k], sizef * n*n/4, cudaMemcpyDeviceToHost )     );

				
#pragma omp parallel for
				for (p=0; p<n/2; p++){
					for (q=0; q<n/2; q++){
						c[ n*(p + gi*n/2) + (q + gj*n/2)  ] += c_aux[ (n/2)*p + q  ];
					}
				}


				/*print_matrix(c);*/

			}


		}
	}


	for (int i=0; i<deviceCount; i++){

		HANDLE_ERROR( cudaSetDevice( i  ));
		
		
		HANDLE_ERROR( cudaEventRecord(mstop[i], 0) );
		HANDLE_ERROR( cudaEventSynchronize( mstop[i]  ));
		
		
		cudaFree( a_mdev[i] );
		cudaFree( b_mdev[i] );
		cudaFree( c_mdev[i] );
	}

	print_matrix(c);

	for (int i=0; i<deviceCount; i++){
		HANDLE_ERROR( cudaSetDevice( i  ));
		float   elapsedTime;
		HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, mstart[i], mstop[i] ) );
		printf( "CPU+mGPU Time: %f  ms\n", elapsedTime );
		printf("speed up: %f \n", (t2-t1)*1000.0/(elapsedTime) );
	}



	
	/*free(a_aux);*/
	/*free(b_aux);*/
	/*free(c_aux);*/

	cudaFreeHost(a_aux);
	cudaFreeHost(b_aux);
	cudaFreeHost(c_aux);







	











	free(a);
	free(b);
	free(c);



	return 0;

}














