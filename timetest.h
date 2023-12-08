#ifndef TIMETEST_H
#define TIMETEST_H

#include <time.h>
#include <stdio.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

static struct timespec time_measure_start;
static struct timespec time_measure_end;

#define TIME_START() \
do { \
    clock_gettime(CLOCK_MONOTONIC, &time_measure_start); \
    printf("[TIME] Measurement started.\n"); \
} while (0)

#define TIME_END() \
do { \
    clock_gettime(CLOCK_MONOTONIC, &time_measure_end); \
    printf("[TIME] Measurement ended.\n"); \
    printf("[TIME] Elapsed time = %f ms\n", ((time_measure_end.tv_sec * 1000000000 + time_measure_end.tv_nsec) - (time_measure_start.tv_sec * 1000000000 + time_measure_start.tv_nsec)) / 1000000.0); \
} while (0)

static cudaEvent_t cuda_measure_start;
static cudaEvent_t cuda_measure_end;
static float cuda_measure_elapsed;

#define CUDA_TIME_START() \
do { \
    cudaEventCreate(&cuda_measure_start);\
    cudaEventCreate(&cuda_measure_end);\
    printf("[TIME] CUDA measurement started.\n"); \
    cudaEventRecord(cuda_measure_start, 0);\
} while (0)

#define CUDA_TIME_END() \
do {\
    cudaEventRecord(cuda_measure_end, 0);\
    printf("[TIME] CUDA measurement ended.\n"); \
    cudaEventSynchronize(cuda_measure_end);\
    cudaEventElapsedTime(&cuda_measure_elapsed, cuda_measure_start, cuda_measure_end);\
    cudaEventDestroy(cuda_measure_start);\
    cudaEventDestroy(cuda_measure_end);\
    printf("[TIME] CUDA elapsed time = %f ms\n", cuda_measure_elapsed);\
} while (0)

#ifdef __cplusplus
}
#endif

#endif // TIMETEST_H
