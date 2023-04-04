#include "./common.cpp"
#include "./image.cpp"

const unsigned image_width = 640;
const unsigned image_height = 400;
const unsigned HIST_SIZE = 32; // histogram levels

int main(const int argc, const char* argv[]) {
    unsigned int* vecHist = (unsigned int*) malloc(HIST_SIZE * sizeof(unsigned int));
    memset(vecHist, 0, HIST_SIZE * sizeof(unsigned int));

    uint32_t* ptr = reinterpret_cast<uint32_t*>(grayscale_data);
    ELAPSED_TIME_BEGIN(0);
    for (int i=0; i<sizeof(grayscale_data)/sizeof(uint32_t); i++) {
        uint32_t val = ptr[i];
        // vecHist[(val && 0xFF) / 8]++;
        // vecHist[((val >> 8) && 0xFF) / 8]++;
        // vecHist[((val >> 16) && 0xFF) / 8]++;
        // vecHist[((val >> 24) && 0xFF) / 8]++;
        for (int i = 0; i < 4; i++) {
            uint32_t convertedVal = (val >> (8*i)) & 0xFF;
            vecHist[(convertedVal & 0xFF) / 8]++;
            // vecHist[convertedVal / (256 / HIST_SIZE)]++;
        }
    }
    ELAPSED_TIME_END(0);

    printf("image pixels = %zu\n", sizeof(grayscale_data));
    printf("histogram levels = %u\n", HIST_SIZE);

    int sum = 0;
    for (int i=0; i<HIST_SIZE; i++) {
        	printf("hist[%2d] = %8u\n", i, vecHist[i]);
        sum+= vecHist[i];
    }

    printf("sum = %u\n", sum);

    free(vecHist);

    return 0;
}