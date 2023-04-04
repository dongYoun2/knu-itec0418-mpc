#include "./common.cpp"

unsigned matsize = 4000; // num rows and also num cols

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
	
	float* matSrc = (float*)malloc(matsize*matsize*sizeof(float));
	float* matDst = (float*)malloc(matsize*matsize*sizeof(float));
	
	srand(0);
	setNormalizedRandomData(matSrc, matsize * matsize);

	ELAPSED_TIME_BEGIN(0);
	for (int y=0; y<matsize; y++) {
		for (int x=0; x<matsize; x++) {
			int srcIdx = y * matsize + x;
			int dstIdx = x * matsize + y;
			matDst[dstIdx] = matSrc[srcIdx];
		}
	}
	ELAPSED_TIME_END(0);
	// check the result
	float sumSrc = getSum( matSrc, matsize * matsize );
	float sumDst = getSum( matDst, matsize * matsize );
	float diff = fabsf( sumDst - sumSrc );
	printf("matrix size = matsize * matsize = %d * %d\n", matsize, matsize);
	printf("sumSrc = %f\n", sumSrc);
	printf("sumDst = %f\n", sumDst);
	printf("diff(sumSrc, sumDst) = %f\n", diff);
	printf("diff(sumSrc, sumDst) / SIZE = %f\n", diff / (matsize * matsize));
	printMat( "matSrc", matSrc, matsize, matsize );
	printMat( "matDst", matDst, matsize, matsize );

	free(matSrc);
	free(matDst);

    return 0;
}
