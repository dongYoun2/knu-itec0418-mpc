#include "./common.cpp"

const int DEVICE_MEM_TYPE_CNT = 6;
const int SHARED_MEM_SIZE_IN_ELEM = 8 * 1024;
unsigned int cnt = 1024 * 1024; // num of samplings

__constant__ float c_value = 1.0f;

__global__ void memSpeedTest(float* global_arr, unsigned int cnt, float* ans, long long* times) {
    clock_t start, end;
    int gx = blockIdx.x * blockDim.x + threadIdx.x;

    // register
    start = clock();
    register float a = 0.0f;
    register float b = 1.0f;
    for (unsigned int i=0; i<cnt; i++) {
        a += b;
    }
    end = clock();
    if (gx==0) {
        ans[0] = a;
        times[0] = (long long)(end - start);
    }

    // shared nemory
    start = clock();
    __shared__ float shMem[SHARED_MEM_SIZE_IN_ELEM];
    shMem[0] = 0.0f;
    for (unsigned int i=0; i<cnt; i++) {
        unsigned nextIdx = (i + 1) % SHARED_MEM_SIZE_IN_ELEM;
        // unsigned prevIdx = (nextIdx + SHARED_MEM_SIZE_IN_ELEM - 1) % SHARED_MEM_SIZE_IN_ELEM;
        unsigned prevIdx = i  % SHARED_MEM_SIZE_IN_ELEM;
        shMem[nextIdx] = shMem[prevIdx] + 1.0f;
    }
    // for (unsigned int j = 0; j < cnt / SHARED_MEM_SIZE_IN_ELEM; j++) {
    //     for (unsigned i = 0; i < SHARED_MEM_SIZE_IN_ELEM; i++) {
    //         unsigned nextIdx = (i + 1) % SHARED_MEM_SIZE_IN_ELEM;
    //         shMem[nextIdx] = shMem[i] + 1.0f;
    //     }
    // }
    end = clock();
    if (gx==0) {
        unsigned int final_idx = cnt % SHARED_MEM_SIZE_IN_ELEM;
        ans[1] = shMem[final_idx];
        times[1] = (long long)(end - start);
    }

    // global memory
    start = clock();
    global_arr[0] = 0.0f;
    for (unsigned int i=0; i<cnt; i++) {
        unsigned int nextIdx = (i + 1) % cnt;
        global_arr[nextIdx] = global_arr[i] + 1;
    }
    end = clock();
    if (gx==0) {
         ans[2] = global_arr[0];
        times[2] = (long long)(end - start);
    }

    // local memory
    start = clock();
    float localMem[SHARED_MEM_SIZE_IN_ELEM];
    localMem[0] = 0.0f;
    for (unsigned int i=0; i<cnt; i++) {
        unsigned nextIdx = (i + 1) % SHARED_MEM_SIZE_IN_ELEM;
        // unsigned prevIdx = (nextIdx + SHARED_MEM_SIZE_IN_ELEM - 1) % SHARED_MEM_SIZE_IN_ELEM;
        unsigned prevIdx = i % SHARED_MEM_SIZE_IN_ELEM;
        localMem[nextIdx] = localMem[prevIdx] + 1.0f;
    }
    // for (unsigned int j = 0; j < cnt / SHARED_MEM_SIZE_IN_ELEM; j++) {
    //     for (unsigned i = 0; i < SHARED_MEM_SIZE_IN_ELEM; i++) {
    //         unsigned nextIdx = (i + 1) % SHARED_MEM_SIZE_IN_ELEM;
    //         localMem[nextIdx] = localMem[i] + 1.0f;
    //     }
    // }
    end = clock();
    if (gx==0) {
        unsigned int final_idx = cnt % SHARED_MEM_SIZE_IN_ELEM;
        ans[3] = localMem[final_idx];
        times[3] = (long long)(end - start);
    }

    // constant memory + register
    start = clock();
    register float c = 0.0f;
    for (unsigned int i=0; i<cnt; i++) {
        c = c + c_value;
    }
    end = clock();
    if (gx==0) {
        ans[4] = c;
        times[4] = (long long)(end - start);
    }
}


int main(const int argc, const char* argv[]) {
    // argv processing
	switch (argc) {
	case 1:
		break;
	case 2:
		cnt = procArg( argv[0], argv[1], 1 );
		break;
	default:
		printf("usage: %s [num]\n", argv[0]);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
		break;
	}
    if (cnt % SHARED_MEM_SIZE_IN_ELEM != 0) {
		printf("CAUTION: cnt=%u is NOT a multiple of SHARED_MEM_SIZE_IN_ELEM=%d.\n", cnt, SHARED_MEM_SIZE_IN_ELEM);
	}
    printf("num=%u, SHMEM_SIZE=%u\n", cnt, SHARED_MEM_SIZE_IN_ELEM);
    float* ans = (float*)malloc(DEVICE_MEM_TYPE_CNT * sizeof(float));
    long long* times = (long long*)malloc(DEVICE_MEM_TYPE_CNT * sizeof(long long));

    float* dev_global_arr = nullptr;
    float* dev_ans = nullptr;
    long long* dev_times = nullptr;

    cudaMalloc(&dev_global_arr, cnt * sizeof(long long));
    CUDA_CHECK_ERROR();
    cudaMalloc(&dev_ans, DEVICE_MEM_TYPE_CNT * sizeof(float));
    CUDA_CHECK_ERROR();
    cudaMalloc(&dev_times, DEVICE_MEM_TYPE_CNT * sizeof(long long));
    CUDA_CHECK_ERROR();

    ELAPSED_TIME_BEGIN(0);
    memSpeedTest<<<1, 1>>>(dev_global_arr, cnt, dev_ans, dev_times);
    cudaDeviceSynchronize();
    ELAPSED_TIME_END(0);
    CUDA_CHECK_ERROR();

    cudaMemcpy(ans, dev_ans, DEVICE_MEM_TYPE_CNT * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR();
    cudaMemcpy(times, dev_times, DEVICE_MEM_TYPE_CNT * sizeof(long long), cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR();

    printf("reg var case:   \tans=%f\tclock=%12lld ticks\n", ans[0], times[0]);
	printf("shared var case:\tans=%f\tclock=%12lld ticks\n", ans[1], times[1]);
	printf("global var case:\tans=%f\tclock=%12lld ticks\n", ans[2], times[2]);
	printf("local var case: \tans=%f\tclock=%12lld ticks\n", ans[3], times[3]);
	printf("const var case: \tans=%f\tclock=%12lld ticks\n", ans[4], times[4]);


    cudaFree(dev_ans);
    CUDA_CHECK_ERROR();
    cudaFree(dev_times);
    CUDA_CHECK_ERROR();
    cudaFree(dev_global_arr);
    CUDA_CHECK_ERROR();

    free(ans);
    free(times);

    return 0;
}