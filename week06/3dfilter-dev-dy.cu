#include "./common.cpp"

dim3 dimImage(300, 300, 256);

__global__ void kernelFilter(float* dst, float* src1, float* src2, dim3 imgShape) {
    unsigned int idx_z = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx_z < imgShape.z && idx_y < imgShape.y && idx_x < imgShape.x) {
        unsigned idx = (idx_z * imgShape.y + idx_y) * imgShape.x + idx_x;
        dst[idx] = src1[idx] * src2[idx];
    }
}

int main(const int argc, const char* argv[]) {
    switch(argc) {
    case 1:
        break;
    case 2:
        dimImage.x = dimImage.y = dimImage.z = procArg<int>(argv[0], argv[1], 32, 1024);
        break;
    case 4:
        dimImage.z = procArg<int>(argv[0], argv[1], 32, 1024);
        dimImage.y = procArg<int>(argv[0], argv[2], 32, 1024);
        dimImage.x = procArg<int>(argv[0], argv[3], 32, 1024);
        break;
    default:
        printf("usage: %s [dim.z [dim.y dim.x]]\n", argv[0]);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
    }
    
    unsigned int matElemCnt = dimImage.z * dimImage.y * dimImage.x;
    size_t mat_size = (size_t)(matElemCnt * sizeof(float));
    float* matA = (float*)malloc(mat_size);
    float* matB = (float*)malloc(mat_size);
    float* matC = (float*)malloc(mat_size);

    srand(0);
    setNormalizedRandomData(matA, matElemCnt);
    setNormalizedRandomData(matB, matElemCnt);

    float *dev_matA = nullptr;
    float *dev_matB = nullptr;
    float *dev_matC = nullptr;

    ELAPSED_TIME_BEGIN(1);

    cudaMalloc(&dev_matA, mat_size);
    CUDA_CHECK_ERROR();
    cudaMalloc(&dev_matB, mat_size);
    CUDA_CHECK_ERROR();
    cudaMalloc(&dev_matC, mat_size);
    CUDA_CHECK_ERROR();

    cudaMemcpy(dev_matA, matA, mat_size, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();
    cudaMemcpy(dev_matB, matB, mat_size, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();

    dim3 dimBlock(8, 8, 8);
    // 이렇게 해도 되지 않나?
    // dim3 dimBlock(8, 8, 16);
    unsigned int grid_x = (dimImage.x + (dimBlock.x - 1)) / dimBlock.x;
    unsigned int grid_y = (dimImage.y + (dimBlock.y - 1)) / dimBlock.y;
    unsigned int grid_z = (dimImage.z + (dimBlock.z - 1)) / dimBlock.z;
    dim3 dimGrid(grid_x, grid_y, grid_z);

    ELAPSED_TIME_BEGIN(0);
    kernelFilter<<<dimGrid, dimBlock>>>(dev_matC, dev_matA, dev_matB, dimImage);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR();
    ELAPSED_TIME_END(0);

    cudaMemcpy(matC, dev_matC, mat_size, cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR();
    ELAPSED_TIME_END(1);

    cudaFree(dev_matA);
    CUDA_CHECK_ERROR();
    cudaFree(dev_matB);
    CUDA_CHECK_ERROR();
    cudaFree(dev_matC);
    CUDA_CHECK_ERROR();

    float sumA = getSum(matA, matElemCnt);
    float sumB = getSum(matB, matElemCnt);
    float sumC = getSum(matC, matElemCnt);
    // float diff = fabsf(sumC - (sumA + sumB));

    printf("matrix size = %d * %d * %d\n", dimImage.z, dimImage.y, dimImage.x);
    printf("sumC: %f\n", sumC);
    printf("sumB: %f\n", sumB);
    printf("sumA: %f\n", sumA);
    // printf("diff(sumC, sumA+sumB): %f", diff);
    // printf("diff(sumC, sumA+sumB) / (dimImage.x*dimImage.y*dimImage.z): %f", diff / matElemCnt);
    print3D("C", matC, dimImage.z, dimImage.y, dimImage.x);
    print3D("A", matA, dimImage.z, dimImage.y, dimImage.x);
    print3D("B", matB, dimImage.z, dimImage.y, dimImage.x);

    free(matA);
    free(matB);
    free(matC);

    return 0;
}