#include <stdio.h>

void printArray(const int* arr, int len, char name) {
    printf("%c: ", name);
    for (int i = 0; i < len; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

void kernelAddition(int i, const int* s1, const int* s2, int* d) {
    d[i] = s1[i] + s2[i];
}

int main(void) {
    const int SIZE = 5;
    const int a[SIZE] = {1,2,3,4,5};
    const int b[SIZE] = {10, 20, 30, 40, 50};
    int c[SIZE] = {0};

    for (int i=0; i<SIZE; i++) {
        kernelAddition(i, a, b, c);
    }

    printArray(a, SIZE, 'a');
    printArray(b, SIZE, 'b');
    printArray(c, SIZE, 'c');

    return 0;
}
