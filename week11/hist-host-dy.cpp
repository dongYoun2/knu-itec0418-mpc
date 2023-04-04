#include "./common.cpp"
#include "./image.cpp"

const unsigned image_width = 640;
const unsigned image_height = 400;
const unsigned HIST_SIZE = 32; // histogram levels

int main(const int argc, const char* argv[]) {
    unsigned int* vecHist = (unsigned int*)malloc(HIST_SIZE*sizeof(unsigned int));
    memset(vecHist, 0, HIST_SIZE*sizeof(unsigned int));

    ELAPSED_TIME_BEGIN(0);
    for (int i=0; i<image_width*image_height; i++) {
        unsigned int grayColor = grayscale_data[i] / (256 / HIST_SIZE);
        assert(0 <= grayColor && grayColor <= 32);
        vecHist[grayColor] += 1;
    }
    ELAPSED_TIME_END(0);
    
    // printf("image pixels = %zu\n", sizeof(grayscale_data));
    printf("image pixels = %u\n", image_width*image_height);
	printf("histogram levels = %u\n", HIST_SIZE);
    
    int sum = 0;
    for (int i=0; i<HIST_SIZE; i++) {
        	printf("hist[%2d] = %8u\n", i, vecHist[i]);
        sum+= vecHist[i];
    }

    printf("sum = %u\n", sum);
    assert(sum == image_width*image_height);

    free(vecHist);

    return 0;
}