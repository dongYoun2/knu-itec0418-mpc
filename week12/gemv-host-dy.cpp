#include "./common.cpp"

const unsigned BLOCK_SIZE = 1024;
unsigned matsize = 16 * 1024; // num rows and also num cols
const float alpha = 0.012;
const float beta = 0.034;

// host-side data: Z = alpha A X + beta Y
float* vecZ = nullptr;
float* matA = nullptr;
float* vecX = nullptr;
float* vecY = nullptr;

void execute(void) {
    ELAPSED_TIME_BEGIN(0);
    for (int y=0; y<matsize; y++) {
        float sum = 0.0f;
        for (int k=0; k<matsize; k++) {
            unsigned int idxA = y * matsize + k;
            sum += matA[idxA] * vecX[k];
        }
        vecZ[y] = alpha * sum + beta * vecY[y];
    }
    ELAPSED_TIME_END(0);
}

void printResult(void) {
    float sumZ = getSum(vecZ, matsize);
    float sumA = getSum(matA, matsize * matsize);
    float sumX = getSum(vecX, matsize);
    float sumY = getSum(vecY, matsize);

    printf("sumZ = %f\n", sumZ);
	printf("sumA = %f\n", sumA);
	printf("sumX = %f\n", sumX);
	printf("sumY = %f\n", sumY);
	printVec( "vecZ", vecZ, matsize );
	printMat( "matA", matA, matsize, matsize );
	printVec( "vecX", vecX, matsize );
	printVec( "vecY", vecY, matsize );
}

int main(const int argc, const char* argv[]) {
    switch (argc) {
    case 1:
        break;
    case 2:
        matsize = procArg<int>(argv[0], argv[1], 4);
        break;
    default:
        printf("usage: %s [matsize]\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    printf("PROBLEM SIZE: matsize = %d * %d\n", matsize, matsize);

    vecZ = (float*)malloc(matsize*sizeof(float));
    matA = (float*)malloc(matsize*matsize*sizeof(float));
    vecX = (float*)malloc(matsize*sizeof(float));
    vecY = (float*)malloc(matsize*sizeof(float));

    srand(0);
    setNormalizedRandomData(matA, matsize*matsize);
    setNormalizedRandomData(vecX, matsize);
    setNormalizedRandomData(vecY, matsize);

    execute();
    printResult();

    free(vecZ);
    free(matA);
    free(vecX);
    free(vecY);

    return 0;
}
