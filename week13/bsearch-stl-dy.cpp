#include "./common.cpp"
#include <algorithm>

using namespace std;

unsigned num = 16 * 1024 * 1024; // maximum num of inputs
unsigned bound = 1000 * 1000; // numbers will be ranged in [0..bound)

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
	printf("num = %d, bound = %d\n", num, bound);

    unsigned* vecData = (unsigned*)malloc(num*sizeof(unsigned));

    srand(0);
    setRandomData<unsigned>(vecData, num, bound);
    unsigned targetValue = vecData[num-1];
    printf("target value = %d\n", targetValue);

    std::sort(vecData, vecData + num);

    ELAPSED_TIME_BEGIN(0);
    bool flag = std::binary_search(vecData, vecData + num, targetValue);
    ELAPSED_TIME_END(0);
    if (flag == false) {
		printf("NOT FOUND: target value '%u' not found\n", targetValue);
	} else {
		printf("FOUND: vecData: %d found\n", targetValue);
	}

    printVec("vecData", vecData, num);

    ELAPSED_TIME_BEGIN(1);
    unsigned* lptr = std::lower_bound(vecData, vecData + num, targetValue);
    unsigned* uptr = std::upper_bound(vecData, vecData + num, targetValue);
    ELAPSED_TIME_END(1);

    // REMIND
    // 못찾았을 때 리턴값 nullptr 아님. 못 찾으면 last를 리턴함. 즉, vecData + num.
    // 이는 주소값임!!
    // if (lptr == nullptr && uptr == nullptr) {
    if (lptr == uptr) {
        printf("NOT FOUND: target value '%u' not found\n", targetValue);
    } else {
        int lower = lptr - vecData;
        int upper = uptr - vecData;
        printf("FOUND: %d elements found\n", upper - lower);
		printf("lower: vecData[%d] = %u\n", lower, vecData[lower]);
		printf("upper: vecData[%d] = %u\n", upper, vecData[upper]);        
    }

    ELAPSED_TIME_BEGIN(2);
    auto pair = std::equal_range(vecData, vecData + num, targetValue);
    ELAPSED_TIME_END(2);
    lptr = pair.first;
    uptr = pair.second;

    // lower_bound(), upper_bound()와 동일!!
    // if (lptr == nullptr && uptr == nullptr) {
    if (lptr == uptr) {
        printf("NOT FOUND: target value '%u' not found\n", targetValue);
    } else {
        int lower = lptr - vecData;
        int upper = uptr - vecData;
        printf("FOUND: %d elements found\n", upper - lower);
		printf("lower: vecData[%d] = %u\n", lower, vecData[lower]);
		printf("upper: vecData[%d] = %u\n", upper, vecData[upper]);        
    }

    free(vecData);

    return 0;
}