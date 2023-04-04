#include "./common.cpp"

// input parameters
unsigned vecSize = 256 * 1024 * 1024; // big-size elements
float lerp_t = 0.234f;

// CUDA kernel function 
__global__ void kernel_lerp(float* z, const float t, const float* x, const float* y, unsigned n) {
    unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        z[i] = fmaf(t, y[i], fmaf(-t, x[i], x[i]));
    }
}

int main(const int argc, const char* argv[]) {
    // argv processing
    switch (argc) {
    case 1:
        break;
    case 2:
        vecSize = procArg(argv[0], argv[1], 1);
        break;
    case 3:
        vecSize = procArg(argv[0], argv[1], 1);
        lerp_t = procArg<float>(argv[0], argv[2]);
        break;
    default:
        printf("usage: %s [num] [a]\n", argv[0]);
		exit( EXIT_FAILURE );
		break;
    }

    // host-side data
    float* vecX = (float*)malloc(sizeof(float)*vecSize);
    float* vecY = (float*)malloc(sizeof(float)*vecSize);
    float* vecZ = (float*)malloc(sizeof(float)*vecSize);

    srand(42);
    setNormalizedRandomData(vecX, vecSize);
    setNormalizedRandomData(vecY, vecSize);

    // device-side data
    float* dev_vecX = nullptr;
    float* dev_vecY = nullptr;
    float* dev_vecZ = nullptr;

    cudaMalloc(&dev_vecX, sizeof(float)*vecSize);
    CUDA_CHECK_ERROR();
    cudaMalloc(&dev_vecY, sizeof(float)*vecSize);
    CUDA_CHECK_ERROR();
    cudaMalloc(&dev_vecZ, sizeof(float)*vecSize);
    CUDA_CHECK_ERROR();

    ELAPSED_TIME_BEGIN(1);
    cudaMemcpy(dev_vecX, vecX, sizeof(float)*vecSize, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();
    cudaMemcpy(dev_vecY, vecY, sizeof(float)*vecSize, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();

    dim3 dimBlock(1024);
    dim3 dimGrid((vecSize + dimBlock.x - 1) / dimBlock.x);
    ELAPSED_TIME_BEGIN(0);
    kernel_lerp<<<dimGrid, dimBlock>>>(dev_vecZ, lerp_t, dev_vecX, dev_vecY, vecSize);
    cudaDeviceSynchronize();
    ELAPSED_TIME_END(0);
    CUDA_CHECK_ERROR();

    cudaMemcpy(vecZ, dev_vecZ, sizeof(float)*vecSize, cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR();
    ELAPSED_TIME_END(1);

    cudaFree(dev_vecX);
    cudaFree(dev_vecY);
    cudaFree(dev_vecZ);

    float sumX = getSum(vecX, vecSize);
    float sumY = getSum(vecY, vecSize);
    float sumZ = getSum(vecZ, vecSize);
    float diff = fabsf(sumZ - ((1.0F - lerp_t) * sumX + lerp_t * sumY));

    printf("SIZE = %d\n", vecSize);
	printf("a    = %f\n", lerp_t);
	printf("sumX = %f\n", sumX);
	printf("sumY = %f\n", sumY);
	printf("sumZ = %f\n", sumZ);
	printf("diff(sumZ, (1-t)*sumX+t*sumY) =  %f\n", diff);
	printf("diff(sumZ, (1-t)*sumX+t*sumY)/SIZE =  %f\n", diff / vecSize);
	printVec( "vecX", vecX, vecSize );
	printVec( "vecY", vecY, vecSize );
	printVec( "vecZ", vecZ, vecSize );

    free(vecX);
    free(vecY);
    free(vecZ);
    
    return 0;
}

