#include "./common.cpp"
#include <algorithm>

const unsigned BLOCK_SIZE = 1024;
const unsigned TOTAL_NUM = 16 * (2 * BLOCK_SIZE);
const unsigned bound = 1000 * 1000;
enum {
	DECREASING = 0,
	INCREASING = 1,
};
unsigned direction = INCREASING;

int compareLess(const void* lhs, const void* rhs) {
    uint32_t l = *(uint32_t*)lhs;
    uint32_t r = *(uint32_t*)rhs;

    return l - r;
}

int compareGreater(const void* lhs, const void* rhs) {
    uint32_t l = *(uint32_t*)lhs;
    uint32_t r = *(uint32_t*)rhs;

    return r - l;
}

int main(const int argc, const char* argv[]) {
    switch (argc) {
	case 1:
		break;
	case 2:
		direction = procArg( argv[0], argv[1], 0, 1 );
		break;
	default:
		printf("usage: %s [direction] with 0=decreasing, 1=increasing\n", argv[0]);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
		break;
	}
	printf("BLOCK_SIZE = %d, TOTAL_NUM = %d\n", BLOCK_SIZE, TOTAL_NUM);
	printf("bound = %d, dir = %s\n", bound, (direction == 0) ? "DECREASING" : "INCREASING" );

    uint32_t* src = (uint32_t*)malloc(TOTAL_NUM * sizeof(uint32_t));
    uint32_t* dst = (uint32_t*)malloc(TOTAL_NUM * sizeof(uint32_t));

    srand(0);
    setRandomData(src, TOTAL_NUM, bound);
    memcpy(dst, src, TOTAL_NUM * sizeof(uint32_t));

    unsigned unitSize = 2 * BLOCK_SIZE;
    assert(TOTAL_NUM % unitSize == 0);
    unsigned numUnits = TOTAL_NUM / unitSize;

    printf("UNIT SIZE = %d\n", unitSize);
	printf("NUM UNITS = %d\n", numUnits);

	ELAPSED_TIME_BEGIN(0);
    if (direction == INCREASING) {
        for (int i = 0; i < numUnits; i++) {
            qsort(dst + i * unitSize, unitSize, sizeof(uint32_t), compareLess);
        }
    } else {
        for (int i = 0; i < numUnits; i++) {
            qsort(dst + i * unitSize, unitSize, sizeof(uint32_t), compareGreater);
        }
    }
    ELAPSED_TIME_END(0);

    printf("%d source units:\n", numUnits);
	for (unsigned i = 0; i < numUnits; ++i) {
		printVec( "src", src + i * unitSize, unitSize );
	}
	printf("%d sorted units:\n", numUnits);
	for (unsigned i = 0; i < numUnits; ++i) {
		printVec( "dst", dst + i * unitSize, unitSize );
	}

    free(src);
    free(dst);

    return 0;
}