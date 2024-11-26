/***************************************************************************************
 * author   : 宋子龙
 * date     : 2024-11-27
 * desp     : common工具，提供错误检测、设置GPU等通用方法。
****************************************************************************************/

#pragma once
#include <stdlib.h>
#include <stdio.h>

// 普通cuda函数错误检测
// 而核函数没有返回值，检测方法为：调用核函数之后使用
/// ErrorCheck(cudaGetLastError(), __FILE__, __LINE__);
cudaError_t ErrorCheck(cudaError_t error_code, const char *filename, int lineNumber) {
    if (error_code != cudaSuccess) {
        printf("CUDA error:\r\ncode=%d, name=%s, description=%s\r\nfile=%s, line%d\r\n",
               error_code, cudaGetErrorName(error_code), cudaGetErrorName(error_code), filename, lineNumber);
        return error_code;
    }
    return error_code;
}

void setGPU() {

    // 检测计算机GPU数量
    int iDeviceCount = 0;
    cudaError_t error = ErrorCheck(cudaGetDeviceCount(&iDeviceCount), __FILE__, __LINE__);

    if (error != cudaSuccess || iDeviceCount == 0) {
        printf("No CUDA campatable GPU found!\n");
        exit(-1);
    } else {
        printf("The count of GPUs is %d.\n", iDeviceCount);
    }

    // 设置GPU设备
    int iDev = 0;
    error = cudaSetDevice(iDev);
    if (error != cudaSuccess) {
        printf("Fail to set GPU 0 for computing");
        exit(-1);
    } else {
        printf("Set GPU 0 for computing.\n");
    }

}
