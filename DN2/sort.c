#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define N 10000 // numbers of elements in array
#define T 4 // number of T
#define QUICK
//#define PRINT


/**
 * @brief Run with `gcc sort.c -lpthread -o sort.out -O2 &&  time ./sort.out`
 * 
 * @param QUICK - optmization for sorting
 * @param PRINT - print numbers
 */

void fill(int* arr, int size);
void print(int* arr, int size);
void * sort(void* args);
void cmp_and_swap(int* left, int* right);
int is_sorted(int* arr, int size);

typedef struct {
    int id;
} Arg;

pthread_barrier_t barrier;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

int nums[N];
pthread_t threads[T];
Arg args[T];
int sorted = 0; // sorted flag
int itt_count = 0;

int main() {
    srand(time(NULL));
    srand(420 * 69); // generate same numbers every time

    fill(nums, N);
    printf("Before sorting........\n");
    print(nums, N);

    pthread_barrier_init(&barrier, NULL, T);
    for (int i = 0; i < T; ++i) {
        args[i].id = i;
        pthread_create(threads + i, NULL, sort, (void*)(args + i));
    }

    for (int i = 0; i < T; ++i) {
        pthread_join(threads[i], NULL);
    }

    printf("Took %d itterations\n", itt_count);
    printf("After sorting.......\n");
    print(nums, N);
    pthread_barrier_destroy(&barrier);

    printf("Sorted: %s\n", is_sorted(nums, N) ? "OK" : "NOT OK!!");
    return 0;
}

void fill(int* arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 100;
    }
}

void print(int* arr, int size) {
    #ifdef PRINT
        printf("[");
        for (int i = 0; i < size; i++) {
            printf("%d", arr[i]);
            if (i < size - 1) {
                printf(", ");
            }

        }
        printf("]\n");
    #endif
}

void * sort(void* args) {
    Arg *arg = (Arg*)args;
    int T_INDEX = arg->id;

    for (int k = 0; k < N; k++) {
        #ifdef QUICK
            if (sorted) break;
            pthread_barrier_wait(&barrier);
        #endif

        for (int j = 2 * T_INDEX; j < N-1; j += 2 * T) {
            cmp_and_swap(nums + j, nums + j + 1);
        }
        pthread_barrier_wait(&barrier);

        
        for (int j = 2 * T_INDEX + 1; j < N-1; j += 2 * T) {
            cmp_and_swap(nums + j, nums + j + 1);
        }
        pthread_barrier_wait(&barrier);

        if (is_sorted(nums, N)) {
            sorted = 1;  // set sorted flag to 1
        }

        pthread_mutex_lock(&lock);
        itt_count++;
        pthread_mutex_unlock(&lock);
    }
}

void cmp_and_swap(int* left, int* right) {
    if (*left > *right) {
        int temp = *left;
        *left = *right;
        *right = temp;
    }
}

int is_sorted(int* arr, int size) {
    int prev = arr[0];
    for (int i = 1; i < N; i++) {
        if (prev > arr[i]) return 0;
        prev = arr[i];
    }

    return 1;
}
