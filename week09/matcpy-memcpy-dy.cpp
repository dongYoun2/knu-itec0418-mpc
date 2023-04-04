#include "./common.cpp"

unsigned int matsize = 4000;

int main(const int argc, const char* argv[]) {
    switch(argc) {
    case 1:
        break;
    case 2:
        matsize = procArg<int>(argv[0], argv[1], 4);
        break;
    default:
        exit(EXIT_FAILURE);
    }

    float* matSrc = (float*)malloc(matsize * matsize * sizeof(float));
    float* matDst = (float*)malloc(matsize * matsize * sizeof(float));

    srand(0);
    setNormalizedRandomData(matSrc, matsize * matsize);

    ELAPSED_TIME_BEGIN(0);
    memcpy(matDst, matSrc, matsize * matsize * sizeof(float));
    ELAPSED_TIME_END(0);

    float sumSrc = getSum(matSrc, matsize * matsize);
    float sumDst = getSum(matDst, matsize * matsize);
    float diff = fabsf(sumSrc - sumDst);

    printf("matrix size: %d * %d\n", matsize, matsize);
    printf("sumSrc: %f\n", sumSrc);
    printf("sumDst: %f\n", sumDst);
    printf("diff(sumSrc, sumDst): %f\n", diff);
    printf("diff(sumSrc, sumDst) / SIZE: %f\n", diff / (matsize * matsize));
    printMat("Src", matSrc, matsize, matsize);
    printMat("Dst", matDst, matsize, matsize);

    free(matSrc);
    free(matDst);

    return 0;
}