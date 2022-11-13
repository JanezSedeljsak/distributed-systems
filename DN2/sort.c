#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define ELEMENTS 6
#define THREADS 1

void fill(int* arr, int size);
void print(int* arr, int size);
void * sort(void* args);
void cmp_and_swap(int* left, int* right);

typedef struct {
    int id;
} Arg;

pthread_barrier_t barrier;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

int nums[ELEMENTS];
pthread_t threads[THREADS];
Arg args[THREADS];

int main() {
    srand(time(NULL));

    // fill elements
    fill(nums, ELEMENTS);
    printf("Before sorting........\n");
    print(nums, ELEMENTS);

    pthread_barrier_init(&barrier, NULL, THREADS);
    int startTemp = 0;
    for (int i = 0; i < THREADS; ++i) {
        args[i].id = i;
        pthread_create(threads + i, NULL, sort, (void*)(args + i));
    }

    for (int i = 0; i < THREADS; ++i) {
        pthread_join(threads[i], NULL);
    }

    printf("After sorting.......\n");
    print(nums, ELEMENTS);
    pthread_barrier_destroy(&barrier);


    return 0;
}

void fill(int* arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 100;
    }
}

void print(int* arr, int size) {
    printf("[");
    for (int i = 0; i < size; i++) {
        printf("%d", arr[i]);
        if (i < size - 1) {
            printf(", ");
        }

    }
    printf("]\n");
}

void * sort(void* args) {
    Arg *arg = (Arg*)args;
    int T_INDEX = arg->id;

    for (int k = 0; k < ELEMENTS; k++) {

        for (int j = 2 * T_INDEX; j < ELEMENTS-1; j += 2 * THREADS) {
            cmp_and_swap(nums + T_INDEX, nums + T_INDEX + 1);
        }
        pthread_barrier_wait(&barrier);

        
        
        for (int j = 2 * T_INDEX + 1; j < ELEMENTS-1; j += 2 * THREADS) {
            cmp_and_swap(nums + T_INDEX, nums + T_INDEX + 1);
        }
        pthread_barrier_wait(&barrier);

        print(nums, ELEMENTS);
    }
}

void cmp_and_swap(int* left, int* right) {
    if (*left > *right) {
        int temp = *left;
        *left = *right;
        *right = temp;
    }
}
