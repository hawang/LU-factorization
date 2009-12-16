#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <cublas.h>
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

__host__ void randmatrix(float* matri, int row, int column){
	float rand_decimal, rand_whole;
	int i,j;
	for(i=0; i < row; i++){
		for(j=0 ; j< column; j++){
			rand_decimal = ((float)(rand() % 100000))/1000000;//random decimals
			rand_whole = (float)(rand() % 100); //random whole numbers from 1-100

			int zero=rand()%100;
			int zer=(zero>90? 0:1);

			matri[i*row+j]=(rand_decimal+rand_whole)
					*pow(-1.0,(rand() % 2)+1)*(zer); //sum of two parts
		}
	}
}

//generate difference matrix val_D, difference ma_A and ma_B
__host__ void  matrixdiff(float* val_D, float* ma_A, float* ma_B, int row, int column){
	int i, j;
	for(i=0; i<row; i++){
		for(j=0; j<column; j++){
			val_D[i*row+j]=ma_A[i*row+j]-ma_B[i*row+j];
		}
	}
}

// returns the frobenius norm of matrix
__host__ float frobenius_norm(float* addr, int row, int column){
	float sum=0;
	
	int i, j;
	for(i=0; i<row; i++){
		for(j=0; j<column; j++){
			sum=sum+pow(addr[i*row+j],(float)2.0);
		}
	}
	return pow(sum,(float) 0.5);
}

//returns residual of two matrix
__host__ float residual(float * A, float *B, int row, int column){
	float macheps=pow((float)2, (float)-13);
	float * diff= (float*)calloc(row*column,sizeof(float));
	matrixdiff(diff,A,B,row,column);
	printf("difference matrix\n");
	printMatrix(diff,row,column);
	float diff_norm=frobenius_norm(diff, row, column);

	float A_norm=frobenius_norm(A,row,column);
	free(diff);
	return (diff_norm/(A_norm*macheps)); 
	
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
		syncthreads();
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

/*
__global__ void MatrixMult(float *A,float *B, float *C){
	extern __shared__ float local[];
	local[threadIdx.x]= A[gridDim.x*blockIdx.x+threadIdx.x]*B[threadIdx.x*blockDim.x+blockIdx.y];
	int i;
	syncthreads();
	 C[blockIdx.x*gridDim.x+blockIdx.y]=0;
	for(i=0;i<blockDim.x ;i++){
		C[blockIdx.x*gridDim.x+blockIdx.y]=C[blockIdx.x*gridDim.x+blockIdx.y]+local[i];	
	}
}*/

int main(int argc, char** argv)
{
	srand(time(0));
	int row=atoi(argv[1]); int column=atoi(argv[2]);
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
//	O_h=(float*)malloc(size);
	Q_h=(float*)malloc(size);
	cudaMalloc((void**) &A_d,size);
	cudaMalloc((void**) &L_d,size);
	cudaMalloc((void**) &C_d,size);
	cudaMalloc((void**) &P_d,size);
	cudaMalloc((void**) &O_d,size);
	cudaMalloc((void**) &Q_d,size);
/*
	A_h[0]=25;  A_h[1]=5;  A_h[2]=1;
	A_h[3]=64;  A_h[4]=8;  A_h[5]=1;
	A_h[6]=144;  A_h[7]=12;  A_h[8]=1;
*/
	randmatrix(A_h, row, column);	
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
//	cudaMemcpy(O_h,O_d,size,cudaMemcpyDeviceToHost);
//	printMatrix(O_h,row,column);

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
	cublasInit();
	cublasSgemm ('T', 'T', row, column,
             column, 1, L_d, row,
             A_d, column, 0,
             C_d, row);

	 cublasSgemm ('T', 'T', row, column,
             column, 1, P_d, row,
             O_d, column, 0,
             Q_d, row);


/*
	MatrixMult<<<grid,column,column>>>(L_d,A_d,C_d);
	MatrixMult<<<grid,column,column>>>(P_d,O_d,Q_d);
*/
	cudaMemcpy(C_h,C_d,size,cudaMemcpyDeviceToHost);
	printf("LU Matrix:\n");
	printMatrix(C_h, row, column);
	printf("\n");
	cudaMemcpy(Q_h,Q_d,size,cudaMemcpyDeviceToHost);
	printf("PA Matrix:\n");
        printMatrix(Q_h, row, column);
        printf("\n");
	printf("the residual of the matrix factorization is %f\n", residual(C_h,Q_h,row,column));

	cublasShutdown();
	free(A_h);free(L_h); cudaFree(A_d); cudaFree(L_d);free(C_h);cudaFree(C_d);free(Q_h); cudaFree(O_d);cudaFree(Q_d);
}
