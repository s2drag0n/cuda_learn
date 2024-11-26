/***************************************************************************************
 * author   : 宋子龙
 * date     : 2024-11-27
 * desp     : 使用cuda程序查看GPU核心数量
 ****************************************************************************************/

#include "tools/common.cuh"

int getSPcores(cudaDeviceProp devProp) {
  int cores = 0;
  int mp = devProp.multiProcessorCount;
  switch (devProp.major) {
  case 2: // Fermi
    if (devProp.minor == 1)
      cores = mp * 48;
    else
      cores = mp * 32;
         break;
  case 3: // Kepler
    cores = mp * 192;
    break;
  case 5: // MaxWell
    cores = mp * 128;
    break;
  case 6: // Pascal
    if ((devProp.minor == 1) || (devProp.minor == 2))
      cores = mp * 128;
    else if (devProp.minor == 0)
      cores = mp * 64;
    else
      printf("Unknown device type\n");
    break;
  case 7: // Volta and Turing
    if ((devProp.minor == 0) || (devProp.minor == 5))
      cores = mp * 64;
    else
      printf("Unknown device type\n");
    break;
  case 8: // Ampere
    if (devProp.minor == 0)
      cores = mp * 64;
    else if (devProp.minor == 6)
      cores = mp * 128;
    else if (devProp.minor == 9)
      cores = mp * 128; // ada lovelace
    else
      printf("Unknown device type\n");
    break;
  case 9: // Hopper
    if (devProp.minor == 0)
      cores = mp * 128;
    else
      printf("Unknown device type\n");
    break;
  default:
    printf("Unknown device type\n");
    break;
  }
  return cores;
}

int main(void) {
  int device_id = 0;
  ErrorCheck(cudaSetDevice(device_id), __FILE__, __LINE__);

  cudaDeviceProp prop;
  ErrorCheck(cudaGetDeviceProperties(&prop, device_id), __FILE__, __LINE__);

  printf("Cpmpute cores count is %d.\n", getSPcores(prop));

  return 0;
}
