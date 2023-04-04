#include "./common.cpp"

unsigned matsize = 4000; // num rows and also num cols

__global__ void kernelMatTranspose(float* dst, float* src, int n, size_t pitchInElem) {
    unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;

    if (gy < n && gx < n) {
        __shared__ float sdata[32][32];
        unsigned int ty = threadIdx.y;
        unsigned int tx = threadIdx.x;
        unsigned int srcIdx = gy * pitchInElem + gx;
        unsigned int dstIdx = gx * pitchInElem + gy;

        sdata[ty][tx] = src[srcIdx];
        // REMIND: 여기에 사실 __syncthreads() 할 필요 없음! 왜냐면 내(thread)가
        // shared Mem. 으로 옮긴 데이터 그대로 내가 global Mem.인 dst에다가 씀!
        // 사실상 이렇게 할꺼면 shared Mem. 굳이 쓸 이유가 없음.
        // warp 단위로 봤을 때 빠른 shared Mem. address 접근을 세로로 하고
        // 느린 global Mem. address 접근을 가로로 하도록 수정해야 shared Mem. 사용하는
        // 의미가 있음!
        __syncthreads();
        // REMIND: 아래 코드 한 줄 주의!! shared Mem.에 올렸다고 해서
        // dst[dstIdx]에 저장할 때 sdata[tx][ty] 되는 거 아님!!!
        // (global Mem. 접근 index(dstIdx)를 swap 했기 때문)
        // BUT, 이렇게 하면 warp 단위로 봤을 때 shared Mem. address는 가로로 접근하지만
        // global Mem. address를 세로로 접근하게 됨.
        // global Mem. 접근을 최대한 빠른 방식으로 (가로로) 하고, shared Mem.을 세로로
        // 하는 것이 더 좋음!
        dst[dstIdx] = sdata[ty][tx];
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

    cudaMallocPitch(&dev_matSrc, &dpitch, matsize * sizeof(float), matsize);
    CUDA_CHECK_ERROR();
    cudaMallocPitch(&dev_matDst, &dpitch, matsize * sizeof(float), matsize);
    CUDA_CHECK_ERROR();

    size_t hostPitch = matsize * sizeof(float);
    ELAPSED_TIME_BEGIN(1);
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