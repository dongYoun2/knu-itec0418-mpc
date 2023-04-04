#include "./common.cpp"

// input parameters
const float alpha = 0.5f;
const float beta = -100.0f;
unsigned matsize = 4096; // num rows and also num cols

int main(const int argc, char* argv[]) {
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

    float* matA = (float*)malloc(matsize * matsize * sizeof(float));
    float* matB = (float*)malloc(matsize * matsize * sizeof(float));
    float* matC = (float*)malloc(matsize * matsize * sizeof(float));
    float* matZ = (float*)malloc(matsize * matsize * sizeof(float));

    srand(0);
    setNormalizedRandomData(matA, matsize * matsize);
    setNormalizedRandomData(matB, matsize * matsize);
    setNormalizedRandomData(matC, matsize * matsize);

    ELAPSED_TIME_BEGIN(0);
    for(int y=0; y<matsize; y++) {
        for(int x=0; x<matsize; x++) {
            float sum = 0.0;
            for(int k=0; k<matsize; k++) {
                unsigned int idxA = y * matsize + k;
                unsigned int idxB = k * matsize + x;
                sum += matA[idxA] * matB[idxB];
            }
            unsigned int idx = y * matsize + x;
            matZ[idx] = alpha * sum + beta * matC[idx];
        }
    }
    ELAPSED_TIME_END(0);

    float sumA = getSum( matA, matsize * matsize );
	float sumB = getSum( matB, matsize * matsize );
	float sumC = getSum( matC, matsize * matsize );
	float sumZ = getSum( matZ, matsize * matsize );
	printf("matrix size = matsize * matsize = %d * %d\n", matsize, matsize);
	printf("sumA = %f\n", sumA);
	printf("sumB = %f\n", sumB);
	printf("sumC = %f\n", sumC);
	printf("sumZ = %f\n", sumZ);
	printMat( "matZ", matZ, matsize, matsize);
	printMat( "matA", matA, matsize, matsize );
	printMat( "matB", matB, matsize, matsize );
	printMat( "matC", matC, matsize, matsize );

    free(matA);
    free(matB);
    free(matC);
    free(matZ);

    return 0;
}