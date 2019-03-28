
#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include "omp.h"


using namespace std;

typedef float bebop;
typedef long int92;




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
	int92 mem =  (n * n * 4 )/(1024*1024);
	printf("Memory required: %ld Mb\n", mem );


	// --------------------- HOST -----------------------
	// Host Data
	bebop *a;
	bebop *b;
	bebop *c;

	t1 = omp_get_wtime();
	a = (bebop *) malloc( sizef*n*n  );
	b = (bebop *) malloc( sizef*n*n );
	c = (bebop *) malloc( sizef*n*n );
	t2 = omp_get_wtime();
	
	printf("Memory malloc Time:%f sec\n",t2-t1);

	// Init Data
	t1 = omp_get_wtime();
	int92 i;	
#pragma omp parallel for
	for (i=0; i<n*n; i++){
		a[i] = (bebop) i;
	}
	t2 = omp_get_wtime();
	
	printf("Init matrix  Time:%f sec  ratio: %f Mb/sec\n",t2-t1, mem/(t2-t1));

	

	/*t1 = omp_get_wtime();*/
	/*[>matmul_host(n,a,b,c);<]*/
	/*t2 = omp_get_wtime();*/
	
	/*printf("CPU ncpu:%d  Time:%f sec\n",ncpu,t2-t1);*/
	for (i=0; i<10; i++){
		printf("a = %f  \n", a[i] );
	}


	// --------------------- END -----------------------

	


	free(a);
	free(b);
	free(c);





	return 0;
}





