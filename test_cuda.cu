#include <stdio.h>

__global__ void hello_from_gpu()
{
    printf("Hello world from the GPU\n");
}

int main(void)
{
  hello_from_gpu<<<2, 2>>>();
  cudaDeviceSynchronize();

  return 0;
}
