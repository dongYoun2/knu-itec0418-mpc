#include "./common.cpp"
#include <stdio.h>

const unsigned SIZE = 1024 * 1024; // 1M elements

// set random value of [0.000, 1.000) to dst array
void setRandomData( float* dst, int size ) {
	while (size--) {
		*dst++ = (rand() % 1000) / 1000.0F;
	}
}

// get total sum of dst array
float getSum( float* dst, int size ) {
	float sum = 0.0F;
	while (size--) {
		sum += *dst++;
	}
	return sum;
}

int main( void ) {
	// host-side data
	float* vecA = (float*)malloc(SIZE * sizeof(float));
	float* vecB = (float*)malloc(SIZE * sizeof(float));
	float* vecC = (float*)malloc(SIZE * sizeof(float));
	// float* vecA = new float[SIZE];
	// float* vecB = new float[SIZE];
	// float* vecC = new float[SIZE];
	// set random data
	srand( 0 );
	setRandomData( vecA, SIZE );
	setRandomData( vecB, SIZE );
	// kernel execution
	chrono::system_clock::time_point time_begin = chrono::system_clock::now();
	for (int i = 0; i < SIZE; ++i) {
		vecC[i] = vecA[i] + vecB[i];
	}
	chrono::system_clock::time_point time_end = chrono::system_clock::now();
	chrono::microseconds time_elapsed_msec = chrono::duration_cast<chrono::microseconds>(time_end - time_begin);
	printf("elapsed wall-clock time = %ld usec\n", (long)time_elapsed_msec.count());
	// check the result
	float sumA = getSum( vecA, SIZE );
	float sumB = getSum( vecB, SIZE );
	float sumC = getSum( vecC, SIZE );
	float diff = fabsf( sumC - (sumA + sumB) );
	printf("SIZE = %d\n", SIZE);
	printf("sumA = %f\n", sumA);
	printf("sumB = %f\n", sumB);
	printf("sumC = %f\n", sumC);
	printf("diff(sumC, sumA+sumB) =  %f\n", diff);
	printf("diff(sumC, sumA+sumB) / SIZE =  %f\n", diff / SIZE);
	printf("vecA = [ %8f %8f %8f %8f ... %8f %8f %8f %8f ]\n",
	       vecA[0], vecA[1], vecA[2], vecA[3], vecA[SIZE - 4], vecA[SIZE - 3], vecA[SIZE - 2], vecA[SIZE - 1]);
	printf("vecB = [ %8f %8f %8f %8f ... %8f %8f %8f %8f ]\n",
	       vecB[0], vecB[1], vecB[2], vecB[3], vecB[SIZE - 4], vecB[SIZE - 3], vecB[SIZE - 2], vecB[SIZE - 1]);
	printf("vecC = [ %8f %8f %8f %8f ... %8f %8f %8f %8f ]\n",
	       vecC[0], vecC[1], vecC[2], vecC[3], vecC[SIZE - 4], vecC[SIZE - 3], vecC[SIZE - 2], vecC[SIZE - 1]);
	// cleaning
	delete[] vecA;
	delete[] vecB;
	delete[] vecC;
	// done
	return 0;
}
