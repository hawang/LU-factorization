#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda.h>
__global__ void LUdecomp(float *A, float* L, int p_r, int p_c, int row){
	float piv=A[p_r*row+p_c];
	if(blockIdx.x>p_r){
		float ratio= A[blockIdx.x*row+p_c]/piv;
		L[blockIdx.x*row+p_c]=ratio;
		A[blockIdx.x*row+threadIdx.x]=A[blockIdx.x*row+threadIdx.x]-(A[p_r*row+threadIdx.x]*ratio);
	}
}

int main(void){
	int row=3; int column=3;
	float* A_h; float * A_d;
	float* L_h; float * L_d;
	size_t size = 9*sizeof(float);
	A_h=(float*)malloc(size);
	L_h=(float*)malloc(size);
	cudaMalloc((void**) &A_d,size);
	cudaMalloc((void**) &L_d,size);
	
	A_h[0]=25;  A_h[1]=5;  A_h[2]=1;
	A_h[3]=64;  A_h[4]=8;  A_h[5]=1;
	A_h[6]=144;  A_h[7]=12;  A_h[8]=1;
	
	int i,j;
	for(i=0;i<row; i++){
		for(j=0; j<column;j++){
			L_h[i*row+j]=(i==j? 1:0);
		}
	}
	
	cudaMemcpy(A_d,A_h,size,cudaMemcpyHostToDevice);
	cudaMemcpy(L_d,L_h,size,cudaMemcpyHostToDevice);

	int k;
	for(k=0;k<row-1;k++)
		LUdecomp<<<row,column>>>(A_d, L_d, k,k,row);
	
	cudaMemcpy(A_h,A_d,size,cudaMemcpyDeviceToHost);
        cudaMemcpy(L_h,L_d,size,cudaMemcpyDeviceToHost);

	printf("U matrix\n");
	for(i=0;i<row; i++){
                for(j=0; j<column;j++){
                       printf("%f ",A_h[i*row+j]);
                }
                printf("\n");
        }
	printf("\n");
	printf("L matrix\n");
	for(i=0;i<row; i++){
                for(j=0; j<column;j++){
                       printf("%f ",L_h[i*row+j]);
                }
		printf("\n");
        }


	free(A_h);free(L_h); cudaFree(A_d); cudaFree(L_d);
}
