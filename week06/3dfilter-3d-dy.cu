#include "common.cpp"

// input parameters
dim3 dimImage( 300, 300, 256 ); // x, y, z order - width (ncolumn), height (nrow), depth

__global__ void kernelFilter(void* dst, void* src1, void* src2, size_t pitch, dim3 shape) {
    unsigned int widthIdx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int heightIdx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int depthIdx = blockIdx.z * blockDim.z + threadIdx.z;

    if (widthIdx < shape.x && heightIdx < shape.y && depthIdx < shape.z) {
        size_t offset = (depthIdx * shape.y + heightIdx) * pitch + widthIdx * sizeof(float);
        *(float*)((char*)dst + offset) = (*(float*)((char*)src1 + offset)) * (*(float*)((char*)src2 + offset));
    }
}

int main(const int argc, const char* argv[]) {
	switch (argc) {
	case 1:
		break;
	case 2:
		dimImage.x = dimImage.y = dimImage.z = procArg<int>( argv[0], argv[1], 32, 1024 );
		break;
	case 4:
		dimImage.z = procArg<int>( argv[0], argv[1], 32, 1024 );
		dimImage.y = procArg<int>( argv[0], argv[2], 32, 1024 );
		dimImage.x = procArg<int>( argv[0], argv[3], 32, 1024 );
		break;
	default:
		printf("usage: %s [dim.z [dim.y dim.x]]\n", argv[0]);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
		break;
	}

    unsigned int totalElemCnt = dimImage.x * dimImage.y * dimImage.z;
    unsigned int matSize = totalElemCnt * sizeof(float);
    float* matA = (float*)malloc(matSize);
    float* matB = (float*)malloc(matSize);
    float* matC = (float*)malloc(matSize);

    srand(0);
    setNormalizedRandomData(matA, totalElemCnt);
    setNormalizedRandomData(matB, totalElemCnt);

    struct cudaExtent extent = make_cudaExtent(dimImage.x * sizeof(float), dimImage.y, dimImage.z);
    struct cudaPitchedPtr dev_pitchedPtrA, dev_pitchedPtrB, dev_pitchedPtrC;

    ELAPSED_TIME_BEGIN(1);
    cudaMalloc3D(&dev_pitchedPtrA, extent);
    CUDA_CHECK_ERROR();
    cudaMalloc3D(&dev_pitchedPtrB, extent);
    CUDA_CHECK_ERROR();
    cudaMalloc3D(&dev_pitchedPtrC, extent);
    CUDA_CHECK_ERROR();

    // REMIND: cudaMemcpy3DParams 구조체 변수 처음 initialize 할 때 {0} 으로
    // 해주는 게 좋음. 왜냐면 해당 구조체의 멤버 변수로 cudaArray_t 타입의
    // 변수도 존재하는데 얘도 전부 0으로 set 해줘야하기 때문.
    struct cudaMemcpy3DParms cpyA = {0};
    struct cudaMemcpy3DParms cpyB = {0};
    size_t host_pitch = dimImage.x * sizeof(float);

    struct cudaPitchedPtr pitchedPtrA = make_cudaPitchedPtr(matA, host_pitch, dimImage.x, dimImage.y);
    struct cudaPitchedPtr pitchedPtrB = make_cudaPitchedPtr(matB, host_pitch, dimImage.x, dimImage.y);
    struct cudaPos zero_pos = make_cudaPos(0, 0, 0);

    cpyA.srcPtr = pitchedPtrA;
    cpyA.srcPos = zero_pos;
    cpyA.dstPtr = dev_pitchedPtrA;
    cpyA.dstPos = zero_pos;
    cpyA.extent = extent;
    cpyA.kind = cudaMemcpyHostToDevice;

    cpyB.srcPtr = pitchedPtrB;
    cpyB.srcPos = zero_pos;
    cpyB.dstPtr = dev_pitchedPtrB;
    cpyB.dstPos = zero_pos;
    cpyB.extent = extent;
    cpyB.kind = cudaMemcpyHostToDevice;

    cudaMemcpy3D(&cpyA);
    CUDA_CHECK_ERROR();
    cudaMemcpy3D(&cpyB);
    CUDA_CHECK_ERROR();

    dim3 dimBlock(8, 8, 8);
    dim3 dimGrid(div_up(dimImage.x, dimBlock.x), div_up(dimImage.y, dimBlock.y), div_up(dimImage.z, dimBlock.z));
    ELAPSED_TIME_BEGIN(0);
    kernelFilter<<<dimGrid, dimBlock>>>(dev_pitchedPtrC.ptr, dev_pitchedPtrA.ptr, dev_pitchedPtrB.ptr, dev_pitchedPtrA.pitch, dimImage);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR();
    ELAPSED_TIME_END(0);

    struct cudaMemcpy3DParms cpyC = {0};
    struct cudaPitchedPtr pitchedPtrC = make_cudaPitchedPtr(matC, host_pitch, dimImage.x, dimImage.y);
    cpyC.srcPtr = dev_pitchedPtrC;
    cpyC.srcPos = zero_pos;
    cpyC.dstPtr = pitchedPtrC;
    cpyC.dstPos = zero_pos;
    cpyC.extent = extent;
    cpyC.kind = cudaMemcpyDeviceToHost;

    cudaMemcpy3D(&cpyC);
    CUDA_CHECK_ERROR();
    ELAPSED_TIME_END(1);

    cudaFree(dev_pitchedPtrA.ptr);
    CUDA_CHECK_ERROR();
    cudaFree(dev_pitchedPtrB.ptr);
    CUDA_CHECK_ERROR();
    cudaFree(dev_pitchedPtrC.ptr);
    CUDA_CHECK_ERROR();

    float sumA = getSum( matA, dimImage.z * dimImage.y * dimImage.x );
	float sumB = getSum( matB, dimImage.z * dimImage.y * dimImage.x );
	float sumC = getSum( matC, dimImage.z * dimImage.y * dimImage.x );
	printf("matrix size = %d * %d * %d\n", dimImage.z, dimImage.y, dimImage.x);
	printf("sumC = %f\n", sumC);
	printf("sumA = %f\n", sumA);
	printf("sumB = %f\n", sumB);
	print3D( "C", matC, dimImage.z, dimImage.y, dimImage.x );
	print3D( "A", matA, dimImage.z, dimImage.y, dimImage.x );
	print3D( "B", matB, dimImage.z, dimImage.y, dimImage.x );

    free(matA);
    free(matB);
    free(matC);

    return 0;
}