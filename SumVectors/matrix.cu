

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "utils.h"

using namespace std;


__global__ void sum_vects(int *a, int *b, int *c  ){

	int g_dimx = gridDim.x;
	int g_dimy = gridDim.y;
	int g_dimz = gridDim.z;


	int b_dimx = blockDim.x; 
	int b_dimy = blockDim.y;
	int b_dimz = blockDim.z;

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int bz = blockIdx.z;	

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int tz = threadIdx.z;


	int idx = tx;

	
	if(idx == 0) printf("CUDA working! \n",idx);
	
	c[idx] = a[idx] + b[idx];

	//printf("gdim %d %d %d - bdim %d %d %d - b %d %d %d - t %d %d %d  \n", g_dimx, g_dimy, g_dimz, b_dimx, b_dimy, b_dimz, bx, by, bz, tx, ty, tz );
}




int main(int argc, char **argv){
	
	cout << "Matrix CUDA\n" << endl;
	
	if ( argc < 1 ){
		cout << "Arguments Error, 1 needed: N" << endl;
		return 0;
	}

	printf("Vector Sum using Dim3  \n");


	int n = atoi( argv[1] );
//	int dim_x = atoi( argv[2] );
//	int dim_y = atoi( argv[3] );


	int *a;
	int *b;
	int *c;


   	a = (int *) malloc(sizeof(int) * n);
	b = (int *) malloc(sizeof(int) * n);
	c = (int *) malloc(sizeof(int) * n);	

	// Fill vectors

	int i;
	for (i = 0; i < n ; i++ ){
		a[i] = i*2;
		b[i] = i*3;
	}
	

	//Print vectors


	for (i = 0; i < n ; i++ ){

		if (i<15){
		printf(" a[%d] = %d \t",i ,a[i]);	
		printf(" b[%d] = %d \n",i ,b[i]);	
		}
	}
	

	
	// CUDA Part

	int *a_dev;
	int *b_dev;
	int *c_dev;

	HANDLE_ERROR(cudaMalloc((void **)&a_dev, sizeof(int) * n ));

	HANDLE_ERROR(cudaMalloc((void **)&b_dev, sizeof(int) * n )) ;

	HANDLE_ERROR(cudaMalloc((void **)&c_dev, sizeof(int) * n )) ;



	HANDLE_ERROR( cudaMemcpy(a_dev , a , sizeof(int) * n ,cudaMemcpyHostToDevice  )   );
	
	HANDLE_ERROR( cudaMemcpy(b_dev , b , sizeof(int) * n, cudaMemcpyHostToDevice  )   );
	
	HANDLE_ERROR( cudaMemcpy(c_dev , c , sizeof(int) * n, cudaMemcpyHostToDevice  )   );


	// Work Threads

	dim3 blocks(1);  //2D dimension blocks
	dim3 threads(n); //2D dimension threads 	

	sum_vects<<<blocks, threads>>>(a_dev , b_dev , c_dev);


	HANDLE_ERROR( cudaMemcpy(c , c_dev , sizeof(int) * n, cudaMemcpyDeviceToHost  )   );

	

	for (i = 0; i < n ; i++ ){
		if (i<15){
		printf(" c[%d] = %d \n",i ,c[i]);	
		}
	}



	return 0;
}



