#include "./common.cpp"
#include <algorithm>
using namespace std;

unsigned num = 16 * 1024 * 1024; // maximum num of inputs
unsigned bound = 1000 * 1000; // numbers will be ranged in [0..bound)

int main(const int argc, const char* argv[]) {
    switch (argc) {
    case 1:
        break;
    case 2:
        num = procArg<int>(argv[0], argv[1], 1024);
        break;
    case 3:
        num = procArg<int>(argv[0], argv[1], 1024);
        bound = procArg<int>(argv[0], argv[1], 1024);
        break;
    default:
        printf("usage: %s [num] [bound]\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    printf("num = %u\n", num);
	printf("bound = %u\n", bound);

    unsigned* vecData = (unsigned*)malloc(num*sizeof(unsigned));
    srand(0);
    setRandomData<unsigned>(vecData, num, bound);

    unsigned* first = vecData;
    unsigned* last = vecData + num;
    unsigned targetValue = vecData[num - 1];
    printf("target value = %u\n", targetValue);

    ELAPSED_TIME_BEGIN(0);
    unsigned* ptr = std::find(first, last, targetValue);
    ELAPSED_TIME_END(0);

    unsigned index = ptr - vecData;

    if (index >= num) {
        printf("NOT FOUND: target value '%u' not found\n", targetValue);
    } else {
        printf("FOUND: vecData[%d] = %d\n", index, *ptr);
        printf("FOUND: vecData[%d] = %d\n", index, vecData[index]);
    }
    printVec("vecData", vecData, num);

    free(vecData);

    return 0;
}