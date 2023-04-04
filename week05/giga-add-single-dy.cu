#include "./common.cpp"

const unsigned int SIZE = 256 * 1024 * 1024;

__global__ void kernelAdd(float* dst, float* src1, float* src2) {
    for (register int i=0; i<SIZE; i++) {
        dst[i] = src1[i] + src2[i];
    }
}

int main(void) {
    float *a, *b, *c;
    a = (float*)malloc(SIZE * sizeof(float));
    b = (float*)malloc(SIZE * sizeof(float));
    c = (float*)malloc(SIZE * sizeof(float));

    srand(42);
    setNormalizedRandomData(a, SIZE);
    setNormalizedRandomData(b, SIZE);

    float *dev_a, *dev_b, *dev_c;
    cudaMalloc(&dev_a, SIZE * sizeof(float));
    CUDA_CHECK_ERROR();
    cudaMalloc(&dev_b, SIZE * sizeof(float));
    CUDA_CHECK_ERROR();
    cudaMalloc(&dev_c, SIZE * sizeof(float));
    
    ELAPSED_TIME_BEGIN(1);
    cudaMemcpy(dev_a, a, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();
    cudaMemcpy(dev_b, b, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();

    dim3 dimGrid(1);
    dim3 dimBlock(1);
    ELAPSED_TIME_BEGIN(0);
    kernelAdd<<<dimGrid, dimBlock>>>(dev_c, dev_a, dev_b);
    cudaDeviceSynchronize();
    ELAPSED_TIME_END(0);
    CUDA_CHECK_ERROR();
    cudaMemcpy(c, dev_c, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    ELAPSED_TIME_END(1);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    CUDA_CHECK_ERROR();

    float sumA = getSum(a, SIZE);
    float sumB = getSum(b, SIZE);
    float sumC = getSum(c, SIZE);
    float diff = fabsf(sumC - (sumA + sumB));

    printf("SIZE = %d\n", SIZE);
	printf("sumA = %f\n", sumA);
	printf("sumB = %f\n", sumB);
	printf("sumC = %f\n", sumC);
	printf("diff(sumC, sumA+sumB) =  %f\n", diff);
	printf("diff(sumC, sumA+sumB) / SIZE =  %f\n", diff / SIZE);
	printVec( "vecA", a, SIZE );
	printVec( "vecB", b, SIZE );
	printVec( "vecC", c, SIZE );
    

    free(a);
    free(b);
    free(c);

    return 0;
}