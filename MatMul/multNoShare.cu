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

  // TODO - load A into device memory
  // declare device matrix, width and height

  // calculate the size of A

  // malloc A on device, size of A

  // memcpy A onto the device

  // TODO - load B into device memory  
  // declare device matrix, width and height

  // calculate the total size

  // malloc B on device (size of B)

  // memcpy A onto the device


  // Allocate C in device memory 
  // declare device matrix, width and height 
  
  // calculate the total size

  // malloc c on the device

  // Set dimBlock, dimGrid

  // Print some dimgrid, dimblock info

  // printf("dimGrid(%i, %i)\n", one, two);

  // TODO - Chevron sytnax launch kernel
  
  // TODO - Read C back from device memory 

  // TODO - Free device memory, (3 calls) 

} 

// Matrix multiplication kernel called by MatMul() 
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) { 
  // Each thread computes one element of C 
  // by accumulating results into Cvalue 

  // TODO zero Cvalue

  // TODO set row

  // TODO set col

  // TODO Compute the new matix C

}

// Usage: multNoShare a1 a2 b2
int main(int argc, char* argv[]){

  // TODO - declare three matracies
	Matrix A, B, C;
	int a1, a2, b1, b2;

// TODO - take inputs from command line
  a1 = atoi(argv[1]);                   /* Height of A */
  a2 = atoi(argv[2]);                   /* Width  of A */
  b1 = a2;                              /* Height of B */
  b2 = atoi(argv[3]);                   /* Width  of B */

  // printf("%i", a1);


  // TODO - Set A heigh width, declare elements
  A.height = a1;
  A.width = a2;
  A.elements = (float*)malloc(A.width * A.height * sizeof(float))\
;

  // TODO - Set B heigh width, declare elements
B.height = b1;
  B.width = b2;
  B.elements = (float*)malloc(B.width * B.height * sizeof(float))\
;

  // TODO - Set C heigh width, declare elements
  C.height = A.height;
  C.width = B.width;
  C.elements = (float*)malloc(C.width * C.height * sizeof(float))\
;
  // TODO - generate some random numbers
  
  // TODO - launch our wrapper program

}
