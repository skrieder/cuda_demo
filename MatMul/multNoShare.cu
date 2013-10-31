/*
 * multNoShare.c
 *
 * Robert Hochberg
 * January 24, 2012
 *
 * Based nearly entirely on the code from the CUDA C Programming Guide
 */

#include "multNoShare.h"

// Matrix multiplication - Host code 
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE 
void MatMul(const Matrix A, const Matrix B, Matrix C) { 

  // Load A and B to device memory 
  Matrix d_A; 
  d_A.width = A.width; 
  d_A.height = A.height; 
  size_t size = A.width * A.height * sizeof(float); 
  cudaError_t err = cudaMalloc(&d_A.elements, size); 
  printf("CUDA malloc A: %s\n",cudaGetErrorString(err)); 
  err = cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice); 
  printf("Copy A to device: %s\n",cudaGetErrorString(err)); 
  
  Matrix d_B; 
  d_B.width = B.width; 
  d_B.height = B.height; 
  size = B.width * B.height * sizeof(float); 
  err = cudaMalloc(&d_B.elements, size); 
  printf("CUDA malloc B: %s\n",cudaGetErrorString(err));
  err = cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
  printf("Copy B to device: %s\n",cudaGetErrorString(err)); 

  // Allocate C in device memory 
  Matrix d_C; 
  d_C.width = C.width; 
  d_C.height = C.height; 
  size = C.width * C.height * sizeof(float); 
  err = cudaMalloc(&d_C.elements, size); 
  printf("CUDA malloc C: %s\n",cudaGetErrorString(err));

  // Invoke kernel 
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); 
  dim3 dimGrid((B.width + dimBlock.x - 1) / dimBlock.x, 
	       (A.height + dimBlock.y - 1) / dimBlock.y); 
  MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C); 
  err = cudaThreadSynchronize();
  printf("Run kernel: %s\n", cudaGetErrorString(err));
  
  // Read C from device memory 
  err = cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost); 
  printf("Copy C off of device: %s\n",cudaGetErrorString(err));

  // Free device memory 
  cudaFree(d_A.elements); 
  cudaFree(d_B.elements); 
  cudaFree(d_C.elements); 
} 

// Matrix multiplication kernel called by MatMul() 
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) { 
  // Each thread computes one element of C 
  // by accumulating results into Cvalue 
  float Cvalue = 0.0; 
  int row = blockIdx.y * blockDim.y + threadIdx.y; 
  int col = blockIdx.x * blockDim.x + threadIdx.x; 
  if(row > A.height || col > B.width) return;
  for (int e = 0; e < A.width; ++e) 
    Cvalue += (A.elements[row * A.width + e]) * (B.elements[e * B.width + col]); 
  C.elements[row * C.width + col] = Cvalue; 
}

// Usage: multNoShare a1 a2 b2
int main(int argc, char* argv[]){
  Matrix A, B, C;
  int a1, a2, b1, b2;
  // Read some values from the commandline
  a1 = atoi(argv[1]);			/* Height of A */
  a2 = atoi(argv[2]);			/* Width  of A */
  b1 = a2;		         	/* Height of B */
  b2 = atoi(argv[3]);			/* Width  of B */

  A.height = a1;
  A.width = a2;
  A.elements = (float*)malloc(A.width * A.height * sizeof(float));

  B.height = b1;
  B.width = b2;
  B.elements = (float*)malloc(B.width * B.height * sizeof(float));

  C.height = A.height;
  C.width = B.width;
  C.elements = (float*)malloc(C.width * C.height * sizeof(float));

  for(int i = 0; i < A.height; i++)
    for(int j = 0; j < A.width; j++)
      A.elements[i*A.width + j] = (float)(rand() % 3);

  for(int i = 0; i < B.height; i++)
    for(int j = 0; j < B.width; j++)
      B.elements[i*B.width + j] = (float)(rand() % 2);

  MatMul(A, B, C);
  /*
  // Print up to a 10x10 portion of the three matrices
  for(int i = 0; i < min(10, A.height); i++){
    for(int j = 0; j < min(10, A.width); j++)
      printf("%f ", A.elements[i*A.width + j]);
    printf("\n");
  }
  printf("\n");

  for(int i = 0; i < min(10, B.height); i++){
    for(int j = 0; j < min(10, B.width); j++)
      printf("%f ", B.elements[i*B.width + j]);
    printf("\n");
  }
  printf("\n");

  for(int i = 0; i < min(10, C.height); i++){
    for(int j = 0; j < min(10, C.width); j++)
      printf("%f ", C.elements[i*C.width + j]);
    printf("\n");
  }
  printf("\n");
  */
}
