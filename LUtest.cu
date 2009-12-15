#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda.h>
//print matrix  function
__host__ void printMatrix(float *A, int row, int column){
	int i,j;
	 for(i=0;i<row; i++){
                for(j=0; j<column;j++){
                       printf("%f ",A[i*row+j]);
                }
                printf("\n");
        }

}

//in this algorithm, the LU decomposition input must be that number of threads = number of columns
// number of blocks be the same as number of rows
__global__ void LUdecomp(float *A, float* L, int p_r, int p_c, float* P){
	float piv=A[p_r*gridDim.x+p_c];
	extern __shared__ float exbuf[]; //row exchange buffer for pivoting 

	if (fabs(piv)<0.00001){
		int row=p_r;int j;double max = piv;
		for(j=p_r; j<gridDim.x;j++){
			if(fabs(A[j*gridDim.x+p_c])>fabs(max)){
				row = j;
				max=A[j*gridDim.x+p_c];
			}
		}
		exbuf[threadIdx.x]= A[p_r*gridDim.x+threadIdx.x];
		A[p_r*gridDim.x+threadIdx.x]=A[row*gridDim.x+threadIdx.x];
		A[row*gridDim.x+threadIdx.x]=exbuf[threadIdx.x];

		exbuf[threadIdx.x]= P[p_r*gridDim.x+threadIdx.x];
		P[p_r*gridDim.x+threadIdx.x]=P[row*gridDim.x+threadIdx.x];
	        P[row*gridDim.x+threadIdx.x]=exbuf[threadIdx.x];
		
		piv=A[p_r*gridDim.x+p_c];	
	}
	
	if(blockIdx.x>p_r){
		float ratio= A[blockIdx.x*gridDim.x+p_c]/piv;
		L[blockIdx.x*gridDim.x+p_c]=ratio;
		A[blockIdx.x*gridDim.x+threadIdx.x]=A[blockIdx.x*gridDim.x+threadIdx.x]
							-(A[p_r*gridDim.x+threadIdx.x]*ratio);
	}
}


__global__ void MatrixMult(float *A,float *B, float *C){
	extern __shared__ float local[];
	local[threadIdx.x]= A[gridDim.x*blockIdx.x+threadIdx.x]*B[threadIdx.x*blockDim.x+blockIdx.y];
	int i;
	syncthreads();
	 C[blockIdx.x*gridDim.x+blockIdx.y]=0;
	for(i=0;i<blockDim.x ;i++){
		C[blockIdx.x*gridDim.x+blockIdx.y]=C[blockIdx.x*gridDim.x+blockIdx.y]+local[i];	
	}
}

int main(void){
	int row=3; int column=3;
	float* A_h; float * A_d;
	float* L_h; float * L_d;
	float* C_h; float * C_d;
	float* P_h; float * P_d;
		    float * O_d;
	float* Q_h; float * Q_d;
	size_t size = row*column*sizeof(float);
	A_h=(float*)malloc(size);
	L_h=(float*)malloc(size);
	C_h=(float*)malloc(size);
	P_h=(float*)malloc(size);
	Q_h=(float*)malloc(size);
	cudaMalloc((void**) &A_d,size);
	cudaMalloc((void**) &L_d,size);
	cudaMalloc((void**) &C_d,size);
	cudaMalloc((void**) &P_d,size);
	cudaMalloc((void**) &O_d,size);
	cudaMalloc((void**) &Q_d,size);

	A_h[0]=25;  A_h[1]=5;  A_h[2]=1;
	A_h[3]=64;  A_h[4]=8;  A_h[5]=1;
	A_h[6]=144;  A_h[7]=12;  A_h[8]=1;
	
	int i,j;
	for(i=0;i<row; i++){
		for(j=0; j<column;j++){
			L_h[i*row+j]=(i==j? 1:0);
			P_h[i*row+j]=(i==j? 1:0);
			C_h[i*row+j]=0;
		}
	}
	
	cudaMemcpy(A_d,A_h,size,cudaMemcpyHostToDevice);
	cudaMemcpy(O_d,A_d,size,cudaMemcpyDeviceToDevice);

	cudaMemcpy(L_d,L_h,size,cudaMemcpyHostToDevice);
	cudaMemcpy(C_d,C_h,size,cudaMemcpyHostToDevice);
	cudaMemcpy(P_d,P_h,size,cudaMemcpyHostToDevice);

	printf("Matrix A\n");
	printMatrix(A_h, row, column);
        printf("\n");

	int k;
	for(k=0;k<row-1;k++)
		LUdecomp<<<row,column,column>>>(A_d, L_d, k,k,P_d);
	
	cudaMemcpy(A_h,A_d,size,cudaMemcpyDeviceToHost);
        cudaMemcpy(L_h,L_d,size,cudaMemcpyDeviceToHost);
	cudaMemcpy(P_h,P_d,size,cudaMemcpyDeviceToHost);

	printf("U matrix\n");
	printMatrix(A_h, row, column);
	printf("\n");
	printf("L matrix\n");
	printMatrix(L_h, row, column);
	printf("\n");
	printf("P matrix\n");
        printMatrix(P_h, row, column);
        printf("\n");


	dim3 grid(row,column);

	MatrixMult<<<grid,column,column>>>(L_d,A_d,C_d);
	MatrixMult<<<grid,column,column>>>(P_d,O_d,Q_d);

	cudaMemcpy(C_h,C_d,size,cudaMemcpyDeviceToHost);
	printf("LU Matrix:\n");
	printMatrix(C_h, row, column);
	printf("\n");
	cudaMemcpy(Q_h,Q_d,size,cudaMemcpyDeviceToHost);
	printf("PA Matrix:\n");
        printMatrix(Q_h, row, column);
        printf("\n");


	free(A_h);free(L_h); cudaFree(A_d); cudaFree(L_d);free(C_h);cudaFree(C_d);free(Q_h); cudaFree(O_d);cudaFree(Q_d);
}
