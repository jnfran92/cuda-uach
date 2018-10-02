


#include <iostream>
#include <stdlib.h>
#include <stdio.h>

void print_dmatrix(double *matrix, int m, int n ){

        printf("Matrix print \n");
        int i,j;
        for ( i =0; i<m ; i++  ){
		for( j=0; j<n; j++ ){
                	
			if(m*i + j < 15 ){
			printf(" matrix[%d][%d] = %f \n", i, j,  matrix[ m*i + j ]);
			}

		}

        }


}





