#include "./common.cpp"

int main(void) {
    unsigned int SIZE = 256 * 1024 * 1024;
    float *a, *b, *c;

    a = (float*)malloc(SIZE * sizeof(float));
    b = (float*)malloc(SIZE * sizeof(float));
    c = (float*)malloc(SIZE * sizeof(float));
    
    srand(42);
    setNormalizedRandomData(a, SIZE);
    setNormalizedRandomData(b, SIZE);

    ELAPSED_TIME_BEGIN(0);
    for(int i=0; i<SIZE; i++) {
        c[i] = a[i] + b[i];
    }
    ELAPSED_TIME_BEGIN(1);

    float sumA = getSum(a, SIZE);
    float sumB = getSum(b, SIZE);
    float sumC = getSum(c, SIZE);
    float diff = fabsf((sumA + sumB) - sumC);
    float meanDiff = diff / SIZE;

    printf("SIZE = %d\n", SIZE);
    printf("sumA = %f\n", sumA);
    printf("sumB = %f\n", sumB);
    printf("sumC = %f\n", sumC);
    printf("diff(sumC, sumA+sumB) = %f\n", diff);
    printf("diff(sumC, sumA+sumB) / SIZE = %f\n", meanDiff);
    printVec("vecA", a, SIZE);
    printVec("vecB", b, SIZE);
    printVec("vecC", c, SIZE);

    free(a);
    free(b);
    free(c);

    return 0;
}