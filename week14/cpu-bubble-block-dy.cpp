#include "./common.cpp"
#include <algorithm>

// input parameters
const unsigned BLOCK_SIZE = 1024;
const unsigned TOTAL_NUM = 16 * (2 * BLOCK_SIZE); // max total num data
const unsigned bound = 1000 * 1000; // numbers will be ranged in [0..bound)
enum {
	DECREASING = 0,
	INCREASING = 1,
};
unsigned direction = INCREASING;

// C style
// template <typename TYPE>
// void compareAndSwap(TYPE* lhs, TYPE* rhs, unsigned dir) {
//     // if dir == 0: decreasing case, we want to make (lhs >= rhs)
// 	// if dir == 1: increasing case, we want to make (lhs < rhs)
// 	if (dir == (*lhs > *rhs)) { // simple swap
// 		TYPE t = *lhs;
// 		*lhs = *rhs;
// 		*rhs = t;
// 	}
// }

template <typename TYPE>
void compareAndSwap(TYPE& lhs, TYPE& rhs, unsigned dir) {
    // if dir == 0: decreasing case, we want to make (lhs >= rhs)
	// if dir == 1: increasing case, we want to make (lhs < rhs)
	if (dir == (lhs > rhs)) { // simple swap
		TYPE t = lhs;
		lhs = rhs;
		rhs = t;
	}
}

template <typename TYPE>
void bubbleSort(TYPE* base, unsigned num, unsigned dir = INCREASING) {
    for (int i = num - 1; i > 0; i--) {
        for (int j = 0; j < i; j++) {
            // compareAndSwap(*(base + j), *(base + (j + 1)), dir);
            compareAndSwap(base[j], base[j + 1], dir);
        }
    }
}

int main(const int argc, const char* argv[]) {
	// argv processing
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

    int unitSize = 2 * BLOCK_SIZE;
    assert(TOTAL_NUM % unitSize == 0);
    int numUnits = TOTAL_NUM / unitSize;

    ELAPSED_TIME_BEGIN(0);
    for (int i = 0; i < numUnits; i++) {
		// direction을 인자로 전달함.
        // typename TYPE 말고는 인자로 전달하는  것이 더 좋은 듯?
        // template 인자로는 variable 전달 못 함. 상수만 가능함.
        bubbleSort(dst + i * unitSize, unitSize, direction);
    }
    ELAPSED_TIME_END(0);

    if (direction == INCREASING) {
		for (unsigned i = 0; i < numUnits; ++i) {
			std::sort( src + i * (2 * BLOCK_SIZE), src + (i + 1) * (2 * BLOCK_SIZE) );
		}
	} else {
		for (unsigned i = 0; i < numUnits; ++i) {
			std::sort( src + i * (2 * BLOCK_SIZE), src + (i + 1) * (2 * BLOCK_SIZE), std::greater<uint32_t>() );
		}
	}

    uint32_t err = getTotalDiff( src, dst, TOTAL_NUM );
	printf("total diff = %d\n", err);
	printf("%d sorted units:\n", numUnits);
	for (unsigned i = 0; i < numUnits; ++i) {
		printVec( "dst", dst + i * unitSize, unitSize );
	}

    free(src);
    free(dst);

    return 0;
}