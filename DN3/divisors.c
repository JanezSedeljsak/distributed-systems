#include <stdio.h>
#include <math.h>
#include <omp.h>

#define MILLION 1000000 
#define N (MILLION)
//#define PRINT

int divsiors_sum(int n);
int divisors[N + 1];

/**
 * @brief 
 * Program is written so that first we build an array of divisors for each number, this way 
 * we get constant look up for each number when looping. In the main loop we use reduction
 * to sum up the "neighbour numbers" and in the end we print the elapsed time and result.
 * 
 * (1 million - found 25275024)
 * Run with: 
 * gcc divisors.c -fopenmp -lm -O2 -o div.out
 * srun --reservation=fri -n1 --cpus-per-task=16 div.out
 * 
 * @result
 * Local: elapsed: 1.26
 * NSC - 1: elapsed: 7.57
 * NSC - 2: elapsed: 4.86
 * NSC - 4: elapsed: 3.15
 * NSC - 8: elapsed: 1.83
 * NSC - 16: elapsed: 0.71
 */

int main()
{
    int start_time = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i <= N; i++)
        divisors[i] = divsiors_sum(i);

    int total = 0;

    #pragma omp parallel for reduction(+:total)
    for (int i = 0; i <= N; i++)
    {
        // get current divisor for example for 220 get 284
        int current_divisor = divisors[i];
        if (i < current_divisor && current_divisor <= N && divisors[current_divisor] == i)
        {
            #ifdef PRINT
                printf("%d %d\n", i, current_divisor);
            #endif

            total += i + current_divisor;
        }
    }

    float elapsed = omp_get_wtime() - start_time;
    printf("Sum of nums: %d, elapsed: %.2f\n", total, elapsed);
    return 0;
}

int divsiors_sum(int n)
{
    int sum = 1;
    int _sqrt = sqrt(n);
    for (int i = 2; i <= _sqrt; i++)
    {
        if (n % i == 0)
            sum += i + n/i;
    }

    return sum;
}