#include <stdio.h>

__global__ void hello( void ) {
	printf( "hello CUDA %d %d !\n",  blockIdx.x, threadIdx.x);
}

int main( void ) {
	hello<<<8,1>>>();
// #if defined(__linux__)
// 	cudaDeviceSynchronize();
// #endif
	fflush( stdout );
	return 0;
}
