/***************************************************************************************
 * author   : 宋子龙
 * date     : 2024-11-27
 * desp     : 检查cuda可用性
****************************************************************************************/

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
