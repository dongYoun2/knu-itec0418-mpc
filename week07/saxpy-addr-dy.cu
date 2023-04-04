#include "./common.cpp"

// fixed parameters
const unsigned vecSize = 64 * 1024 * 1024; // big-size elements
const float host_a = 1.234f;
__device__ float dev_a;

float vecA[vecSize];
float vecB[vecSize];
float vecC[vecSize];

__device__ float dev_vecA[vecSize];
__device__ float dev_vecB[vecSize];
__device__ float dev_vecC[vecSize];

__global__ void kernelSAXPY(unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) {
        dev_vecC[idx] = dev_a * dev_vecA[idx] + dev_vecB[idx];
    }
}


int main(const int argc, const char* argv[]) {
    // argv processing
	switch (argc) {
	case 1:
		break;
	default:
		printf("usage: %s\n", argv[0]); // everything fixed !
		exit( EXIT_FAILURE );
		break;
	}

    srand(0);
    setNormalizedRandomData(vecA, vecSize);
    setNormalizedRandomData(vecB, vecSize);

    float* dev_vecA_ptr = nullptr;
    float* dev_vecB_ptr = nullptr;
    float* dev_a_ptr = nullptr;

    ELAPSED_TIME_BEGIN(1);

    cudaGetSymbolAddress((void**)&dev_a_ptr, dev_a);
    CUDA_CHECK_ERROR();
    cudaGetSymbolAddress((void**)&dev_vecA_ptr, dev_vecA);
    CUDA_CHECK_ERROR();
    cudaGetSymbolAddress((void**)&dev_vecB_ptr, dev_vecB);
    CUDA_CHECK_ERROR();
    cudaMemcpy(dev_a_ptr, &host_a, 1 * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();
    cudaMemcpy(dev_vecA_ptr, vecA, vecSize * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();
    cudaMemcpy(dev_vecB_ptr, vecB, vecSize * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();

    dim3 dimBlock(1024);
    dim3 dimGrid(div_up(vecSize, dimBlock.x));
    ELAPSED_TIME_BEGIN(0);
    kernelSAXPY<<<dimGrid, dimBlock>>>(vecSize);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR();
    ELAPSED_TIME_END(0);

    void* dev_c_ptr = nullptr;
    cudaGetSymbolAddress(&dev_c_ptr, dev_vecC);
    CUDA_CHECK_ERROR();
    cudaMemcpy(vecC, dev_c_ptr, vecSize * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR();
    ELAPSED_TIME_END(1);

    float sumX = getSum( vecA, vecSize );
	float sumY = getSum( vecB, vecSize );
	float sumZ = getSum( vecC, vecSize );
	float diff = fabsf( sumZ - (host_a * sumX + sumY) );
	printf("SIZE = %d\n", vecSize);
	printf("a    = %f\n", host_a);
	printf("sumX = %f\n", sumX);
	printf("sumY = %f\n", sumY);
	printf("sumZ = %f\n", sumZ);
	printf("diff(sumZ, a*sumX+sumY) =  %f\n", diff);
	printf("diff(sumZ, a*sumX+sumY)/SIZE =  %f\n", diff / vecSize);
	printVec( "vecX", vecA, vecSize );
	printVec( "vecY", vecB, vecSize );
	printVec( "vecZ", vecC, vecSize );

    return 0;
}