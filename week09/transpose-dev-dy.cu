#include "./common.cpp"

unsigned matsize = 4000; // num rows and also num cols

__global__ void kernelMatTranspose(float* dst, float* src, int n, size_t pitchInElem) {
    unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;

    if (gy < n && gx < n) {
        unsigned int srcIdx = gy * pitchInElem + gx;
        unsigned int dstIdx = gx * pitchInElem + gy;
        dst[dstIdx] = src[srcIdx];
    }
}

int main(const int argc, const char* argv[]) {
	// argv processing
	switch (argc) {
	case 1:
		break;
	case 2:
		matsize = procArg( argv[0], argv[1], 4 );
		break;
	default:
		printf("usage: %s [matsize]\n", argv[0]);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
		break;
	}

    float* matSrc = (float*)malloc(matsize * matsize * sizeof(float));
    float* matDst = (float*)malloc(matsize * matsize * sizeof(float));

    srand(0);
    setNormalizedRandomData(matSrc, matsize * matsize);

    float* dev_matSrc = nullptr;
    float* dev_matDst = nullptr;
    size_t dpitch = 0;

    ELAPSED_TIME_BEGIN(1);
    cudaMallocPitch(&dev_matSrc, &dpitch, matsize * sizeof(float), matsize);
    CUDA_CHECK_ERROR();
    cudaMallocPitch(&dev_matDst, &dpitch, matsize * sizeof(float), matsize);
    CUDA_CHECK_ERROR();

    size_t hostPitch = matsize * sizeof(float);
    cudaMemcpy2D(dev_matSrc, dpitch, matSrc, hostPitch, matsize * sizeof(float), matsize, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();

    dim3 dimBlock(32, 32);
    dim3 dimGrid(div_up(matsize, dimBlock.x), div_up(matsize, dimBlock.y));
    assert(dpitch % sizeof(float) == 0);
    size_t pitchInElem = dpitch / sizeof(float);
    ELAPSED_TIME_BEGIN(0);
    kernelMatTranspose<<<dimGrid, dimBlock>>>(dev_matDst, dev_matSrc, matsize, pitchInElem);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR();
    ELAPSED_TIME_END(0);

    cudaMemcpy2D(matDst, hostPitch, dev_matDst, dpitch, matsize * sizeof(float), matsize, cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR();
    ELAPSED_TIME_END(1);

    cudaFree(dev_matSrc);
    CUDA_CHECK_ERROR();
    cudaFree(dev_matDst);
    CUDA_CHECK_ERROR();

    float sumSrc = getSum( matSrc, matsize * matsize );
	float sumDst = getSum( matDst, matsize * matsize );
	float diff = fabsf( sumDst - sumSrc );
	printf("matrix size = matsize * matsize = %d * %d\n", matsize, matsize);
	printf("sumSrc = %f\n", sumSrc);
	printf("sumDst = %f\n", sumDst);
	printf("diff(sumSrc, sumDst) = %f\n", diff);
	printf("diff(sumSrc, sumDst) / SIZE = %f\n", diff / (matsize * matsize));
	printMat( "Src", matSrc, matsize, matsize );
	printMat( "Dst", matDst, matsize, matsize );

    free(matSrc);
    free(matDst);

    return 0;
}