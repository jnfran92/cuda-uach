

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "../Utils/utils.h"
#include "../Utils/matutils.h"
#include <math.h>       /* sqrt */

#include "csvfile.h"


#include "omp.h"


#define BSIZE 32


//using namespace std;


typedef struct {
	double re;
	double im;
} Complex;


typedef struct {
	Complex c;
	double *field;
} JuliaObj;


void save_julia_data(double *matrix, int m, int n){

	try {
		csvfile csv("./Data/JuliaSet2.csv"); // throws exceptions!

		csv << "x coord";
		csv << "y coord";
		csv << "z coord";
		csv << "scalar";
		csv << endrow;
		/*int i,j;*/

		for (int i = 0; i < m; i++) {

			for (int j=0 ; j<n ; j++){
				csv << j; //x
				csv << i; //y
				csv << 0; //z
				csv << matrix[m*i + j];
				csv << endrow;

			}
			/*csv << endrow;*/

		}


	} catch (const std::exception &ex) {
		std::cout << "Exception was thrown: " << ex.what() << std::endl;
	}




}




__global__ void julia_kernel(int n,  JuliaObj juliaObj){

	int i = threadIdx.y;
	int j = threadIdx.x;

	int gi = threadIdx.y + blockDim.y*blockIdx.y;
	int gj = threadIdx.x + blockDim.x*blockIdx.x;


	int g_dimj = gridDim.x*blockDim.x/2;
	int g_dimi = gridDim.y*blockDim.y/2;


	const double scale = 0.8;

	double coor_i = scale *  (double) (gi - g_dimi)/g_dimi;
	double coor_j = scale *  (double) (gj - g_dimj)/g_dimj;


	/*printf("%f -  %f \n ",coor_i,coor_j   );*/

	Complex z;
	Complex c;

	c = juliaObj.c;


	z.re = coor_j;
	z.im = coor_i;

	int k;

	/*int speed = 0 ;*/

	for(k=0;k<n;k++){

		double re = z.re*z.re - z.im*z.im + c.re;
		double im = 2*z.re*z.im + c.im;

		z.re = re;
		z.im = im;

		double z_mod2 = sqrt( z.re*z.re + z.im*z.im);
		/*printf(" z_mod2  = %f\n ",z_mod2);*/
		if (z_mod2 >99999999999999999999 ){
		/*printf("zmod2  %f  %d \n",z_mod2,k+1);*/
			/*speed = k + 1;*/
			break;
		}

		/*speed = z_mod2;*/

	}


	juliaObj.field[ gridDim.y*blockDim.y*gi + gj ] = k;



}



int main( int argc, char**  argv  ){

	int args_needed = 4;
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


	printf(" CUDA - Julia \n");

	//Init parameters

	int n = atoi(argv[1]); // Convergence maximum
	printf( "n = %d\n",n  );
	int width = atoi(argv[2]);
	int height = atoi(argv[3]);

	// Cretae julia Object

	double *field;

	// Host Data

	field = (double *) malloc( sizeof(double) * width * height );

	init_dmatrix_zeros( field, width, height);

	// Device Data

	Complex c;
	c.re = -0.8;
	c.im = 0.156;


	JuliaObj juliaObj_dev;

	juliaObj_dev.c = c;

	HANDLE_ERROR( cudaMalloc((void **)&juliaObj_dev.field, sizeof(double) * width * height)    );



	// Copy Data to Device


	HANDLE_ERROR( cudaMemcpy(juliaObj_dev.field, field, sizeof(double) * width * height , cudaMemcpyHostToDevice   )  );


	/*print_dmatrix(field  ,width, height);*/

	// Kernel Implementation

	float ms = 0.0;

	dim3 block(BSIZE, BSIZE,1);
	dim3 grid(height/BSIZE, width/BSIZE,1);

	cudaEventRecord(start);
	julia_kernel<<<grid, block>>>(n, juliaObj_dev);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);


	// Retrieve Data from Device
	// Get data Devices
	HANDLE_ERROR(  cudaMemcpy(field, juliaObj_dev.field, sizeof(double) * width * height, cudaMemcpyDeviceToHost )     );


	/*print_dmatrix(field, height, width);*/

	/*save_julia_data( field, height, width  );*/


	double thrs = (double)atoi(argv[4]);
	int x,y;
	for (y = 0; y<height; y++){
		for(x=0; x<width; x++){

			if(field[height*y + x] <= thrs ){
				printf(" ");
			}else{
				printf(".");
			}
		}

		printf("\n");
	}







	ms = 0;
	cudaEventElapsedTime(&ms, start, stop);
	printf("GPU Time: %f\n", ms );


	// Free memory

	cudaFree( juliaObj_dev.field );

	free(field);





	return 0;
}





