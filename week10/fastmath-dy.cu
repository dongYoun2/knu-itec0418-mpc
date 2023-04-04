#include "./common.cpp"
#define _USE_MATH_DEFINES
#include <math.h>

unsigned cnt = 1024 * 1024; // num of samplings

void hostMathTest(float* ans, unsigned cnt) {
    float theta = (2.0f * M_PI) / cnt;
    float acc = 0.0;
    for (unsigned int i=0; i<cnt; i++) {
        float r = theta * i;
        acc += sinf(r) * sinf(r) + cos(r) * cos(r);
    }
    ans[0] = acc;
}

__global__ void kernelMathTest(float* ans, unsigned cnt) {
    float theta = (2.0f * M_PI) / cnt;
    float acc = 0.0;
    for (unsigned int i=0; i<cnt; i++) {
        float r = theta * i;
        acc += sinf(r) * sinf(r) + cos(r) * cos(r);
    }
    ans[1] = acc;
}

__global__ void kernelFastMathTest(float* ans, unsigned cnt) {
    float theta = (2.0f * M_PI) / cnt;
    float acc = 0.0;
    for (unsigned int i=0; i<cnt; i++) {
        float r = theta * i;
        acc += __sinf(r) * __sinf(r) + __cosf(r) * __cosf(r);
    }
    ans[2] = acc;
}

__global__ void kernelFastSinCos(float* ans, unsigned cnt) {
    float theta = (2.0f * M_PI) / cnt;
    float acc = 0.0;
    for (unsigned int i=0; i<cnt; i++) {
        float r = theta * i;
        float sine, cosine;
        __sincosf(r, &sine, &cosine);
        acc += sine * sine + cosine * cosine;
    }
    ans[3] = acc;
}

__global__ void kernelFastSinCosFma(float* ans, unsigned cnt) {
    float theta = (2.0f * M_PI) / cnt;
    float acc = 0.0;
    for (unsigned int i=0; i<cnt; i++) {
        float r = theta * i;
        float sine, cosine;
        __sincosf(r, &sine, &cosine);
        // REMIND: 결과 값 리턴해주는 거 잊지말기!
        acc = __fma_rn(sine, sine, acc);
        acc = __fma_rn(cosine, cosine, acc);
    }
    ans[4] = acc;
}

int main(const int argc, char* argv[]) {
    // argv processing
	switch (argc) {
	case 1:
		break;
	case 2:
		cnt = procArg( argv[0], argv[1], 1 );
		break;
	default:
		printf("usage: %s [cnt]\n", argv[0]);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
		break;
	}
	printf("cnt=%d\n", cnt);

    // float* answers = (float*)malloc(5 * sizeof(float));
    float answers[5];
    float* dev_answers = nullptr;
    cudaMalloc(&dev_answers, 5 * sizeof(float));
    CUDA_CHECK_ERROR();

    ELAPSED_TIME_BEGIN(0);
    hostMathTest(answers, cnt);
    ELAPSED_TIME_END(0);

    cudaMemcpy(dev_answers, answers, 5 * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();

    ELAPSED_TIME_BEGIN(1);
    kernelMathTest<<<1, 1>>>(dev_answers, cnt);
    cudaDeviceSynchronize();
    ELAPSED_TIME_END(1);
    CUDA_CHECK_ERROR();

    ELAPSED_TIME_BEGIN(2);
    kernelFastMathTest<<<1, 1>>>(dev_answers, cnt);
    cudaDeviceSynchronize();
    ELAPSED_TIME_END(2);
    CUDA_CHECK_ERROR();

    ELAPSED_TIME_BEGIN(3);
    kernelFastSinCos<<<1, 1>>>(dev_answers, cnt);
    cudaDeviceSynchronize();
    ELAPSED_TIME_END(3);
    CUDA_CHECK_ERROR();

    ELAPSED_TIME_BEGIN(4);
    kernelFastSinCosFma<<<1, 1>>>(dev_answers, cnt);
    cudaDeviceSynchronize();
    ELAPSED_TIME_END(4);
    CUDA_CHECK_ERROR();

    cudaMemcpy(answers, dev_answers, 5 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_answers);
    CUDA_CHECK_ERROR();

    printf("host math:\tresult=%f\n", answers[0]);
	printf("cuda math:\tresult=%f\n", answers[1]);
	printf("fast math:\tresult=%f\n", answers[2]);
	printf("sincos   :\tresult=%f\n", answers[3]);
	printf("sincosfma:\tresult=%f\n", answers[4]);

    // free(answers);
    return 0;
}