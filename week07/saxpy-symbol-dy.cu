#include "./common.cpp"

// fixed parameters
const unsigned vecSize = 64 * 1024 * 1024; // big-size elements
const float host_a = 1.234f;
__constant__ float dev_a = 1.234f;

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

    ELAPSED_TIME_BEGIN(1);
    cudaMemcpyToSymbol(dev_vecA, vecA, vecSize * sizeof(float));
    CUDA_CHECK_ERROR();
    cudaMemcpyToSymbol(dev_vecB, vecB, vecSize * sizeof(float));
    CUDA_CHECK_ERROR();

    dim3 dimBlock(1024);
    dim3 dimGrid(div_up(vecSize, dimBlock.x));
    ELAPSED_TIME_BEGIN(0);
    kernelSAXPY<<<dimGrid, dimBlock>>>(vecSize);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR();
    ELAPSED_TIME_END(0);

    // REMIND: 포인터 아니고 배열로 선언했기 때문에 cudaMemcpyFromSymbol( host_z, dev_z, sizeof(host_z)) 로 해도 됨!
    cudaMemcpyFromSymbol(vecC, dev_vecC, vecSize * sizeof(float));
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