#include "./common.cpp"
#include <algorithm>
using namespace std;

unsigned num = 16 * 1024 * 1024;
unsigned bound = 1000 * 1000;

int main(const int argc, const char* argv[]) {
    // argv processing
	switch (argc) {
	case 1:
		break;
	case 2:
		num = procArg( argv[0], argv[1], 1024 );
		break;
	case 3:
		num = procArg( argv[0], argv[1], 1024 );
		bound = procArg( argv[0], argv[2], 1024 );
		break;
	default:
		printf("usage: %s [num] [bound]\n", argv[0]);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
		break;
	}
	printf("num = %u\n", num);
	printf("bound = %u\n", bound);

    unsigned* vecData = (unsigned*)malloc(num*sizeof(unsigned));
    const unsigned arrSize = (num / bound) * 4;
    unsigned indices[arrSize];

    srand(0);
    setRandomData<unsigned>(vecData, num, bound);
    unsigned targetValue = vecData[num - 1];
    printf("target value = %u\n", targetValue);

    unsigned* beginPtr = nullptr;
    unsigned* endPtr = nullptr;
    unsigned * findPtr = nullptr;
    unsigned next = 0;
    ELAPSED_TIME_BEGIN(0);
    for (beginPtr = vecData, endPtr = vecData + num; beginPtr < endPtr; beginPtr = findPtr + 1) {
        findPtr = std::find(beginPtr, endPtr, targetValue);
        if (findPtr < endPtr) {
            indices[next++] = findPtr - vecData;
        }
    }

    ELAPSED_TIME_END(0);

    printf("%d locations are found\n", next);
    for (unsigned i = 0; i < next; i++) {
		unsigned index = indices[i];
		printf("vecData[%d]= %d\n", index, vecData[index]);
	}
	printVec( "indices", indices, next );
	printVec( "vecData", vecData, num );

    free(vecData);

    return 0;
}