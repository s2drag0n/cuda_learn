/***************************************************************************************
 * author   : 宋子龙
 * date     : 2024-11-27
 * desp     : 记录cuda程序运行时间
****************************************************************************************/

#include "tools/common.cuh"
#include <cstdio>

__device__ float add(const float x, const float y) {
    return x + y;
}

__global__ void addFromGPU(float *A, float *B, float *C, const int N) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int id = tid + bid * blockDim.x;

    if (id >= N) return;

    C[id] = add(A[id], B[id]);
}

void initialData(float *addr, int elemCount) {
    for (int i = 0; i < elemCount; ++i) {
        addr[i] = (float)(rand() & 0xFF) / 10.f;
    }
    return;
}

int main(void) {
    // 1.设置GPU设备
    setGPU();

    // 2.分配主机和设备内存
    int iElemCount = 512;
    size_t stBytesCount = iElemCount * sizeof(float);

    // 2.1分配主机内存，并初始化
    float *fpHost_A, *fpHost_B, *fpHost_C;
    fpHost_A = (float *)malloc(stBytesCount);
    fpHost_B = (float *)malloc(stBytesCount);
    fpHost_C = (float *)malloc(stBytesCount);
    if (fpHost_A != nullptr && fpHost_B != nullptr && fpHost_C != nullptr) {
        memset(fpHost_A, 0, stBytesCount);
        memset(fpHost_B, 0, stBytesCount);
        memset(fpHost_C, 0, stBytesCount);
    } else {
        printf("Fail to allocate host memory!\n");
        exit(-1);
    }

    // 2.2分配设备内存，并初始化
    float *fpDevice_A, *fpDevice_B, *fpDevice_C;
    ErrorCheck(cudaMalloc((float **)&fpDevice_A, stBytesCount), __FILE__, __LINE__);
    ErrorCheck(cudaMalloc((float **)&fpDevice_B, stBytesCount), __FILE__, __LINE__);
    ErrorCheck(cudaMalloc((float **)&fpDevice_C, stBytesCount), __FILE__, __LINE__);
    if (fpDevice_A != nullptr && fpDevice_B != nullptr && fpDevice_C != nullptr) {
        ErrorCheck(cudaMemset(fpDevice_A, 0, stBytesCount), __FILE__, __LINE__);
        ErrorCheck(cudaMemset(fpDevice_B, 0, stBytesCount), __FILE__, __LINE__);
        ErrorCheck(cudaMemset(fpDevice_C, 0, stBytesCount), __FILE__, __LINE__);
    } else {
        printf("Fail to allocate device memory!\n");
        free(fpHost_A);
        free(fpHost_B);
        free(fpHost_C);
        exit(-1);
    }

    // 3.初始化主机中的数据
    srand(666);
    initialData(fpHost_A, iElemCount);
    initialData(fpHost_B, iElemCount);

    // 4.数据从主机复制到设备
    ErrorCheck(cudaMemcpy(fpDevice_A, fpHost_A, iElemCount, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    ErrorCheck(cudaMemcpy(fpDevice_B, fpHost_A, iElemCount, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    ErrorCheck(cudaMemcpy(fpDevice_C, fpHost_A, iElemCount, cudaMemcpyHostToDevice), __FILE__, __LINE__);

    // 5.调用核函数在设备中进行计算
    dim3 block(32);
    dim3 grid((iElemCount + block.x - 1) / 32);

    int NUM_REPEATS = 10;
    float t_sum = 0;

    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat) {

        cudaEvent_t start, stop;
        ErrorCheck(cudaEventCreate(&start), __FILE__, __LINE__);
        ErrorCheck(cudaEventCreate(&stop), __FILE__, __LINE__);
        ErrorCheck(cudaEventRecord(start), __FILE__, __LINE__);
        cudaEventQuery(start); // 此处不可用错误检测函数

        addFromGPU<<<grid, block>>>(fpDevice_A, fpDevice_B, fpDevice_C, iElemCount);
        ErrorCheck(cudaGetLastError(), __FILE__, __LINE__);

        ErrorCheck(cudaEventRecord(stop), __FILE__, __LINE__);
        ErrorCheck(cudaDeviceSynchronize(), __FILE__, __LINE__);

        float elapsed_time;
        ErrorCheck(cudaEventElapsedTime(&elapsed_time, start, stop), __FILE__, __LINE__);

        if (repeat > 0) {
            t_sum += elapsed_time;
        }

        ErrorCheck(cudaEventDestroy(start), __FILE__, __LINE__);
        ErrorCheck(cudaEventDestroy(stop), __FILE__, __LINE__);
    }

    const float t_ave = t_sum / NUM_REPEATS;
    printf("Time = %g ms.\n", t_ave);


    // 6.将计算得到的数据从设备传给主机
    ErrorCheck(cudaMemcpy(fpHost_C, fpDevice_C, stBytesCount, cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    // for (int i = 0; i < 10; ++i) {
    //     printf("idx=%2d\tmatrix_A:%.2f\tmatrix_B:%.2f\tresult=%.2f\n", i + 1, fpHost_A[i], fpHost_B[i], fpHost_C[i]);
    // }

    // 7.释放主机与设备内存
    free(fpHost_A);
    free(fpHost_B);
    free(fpHost_C);
    ErrorCheck(cudaFree(fpDevice_A), __FILE__, __LINE__);
    ErrorCheck(cudaFree(fpDevice_B), __FILE__, __LINE__);
    ErrorCheck(cudaFree(fpDevice_C), __FILE__, __LINE__);
    
    ErrorCheck(cudaDeviceReset(), __FILE__, __LINE__);

    return 0;
}
