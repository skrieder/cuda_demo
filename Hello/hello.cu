#include <stdio.h>
 
const int N = 16; 
const int blocksize = 16; 
 
__global__ 
void hello(char *ad, int *bd) 
{
  int tidx = blockIdx.x;
  int tidy = blockIdx.y;
  int tidz = threadIdx.x;
  printf("TID = <%d,%d,%d>\n", tidx, tidy, tidz);
}
 
int main()
{
  char a[N] = "Hello \0\0\0\0\0\0";
  int b[N] = {15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
 
  char *ad;
  int *bd;
  const int csize = N*sizeof(char);
  const int isize = N*sizeof(int);
 
  printf("%s", a);
 
  cudaMalloc( (void**)&ad, csize ); 
  cudaMalloc( (void**)&bd, isize ); 
  cudaMemcpy( ad, a, csize, cudaMemcpyHostToDevice ); 
  cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice ); 
  
  dim3 dimGrid( 25, 25 );
  dim3 dimBlock( 10, 10 );
  hello<<<dimGrid, dimBlock>>>(ad, bd);
  cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost ); 
  cudaFree( ad );
  cudaFree( bd );

  printf("%s\n", a);
  return EXIT_SUCCESS;
}
