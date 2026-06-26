#include <omp.h>
#include <iostream>

void evaluate_serial(const int* heights, const int* holes, const int* lines, 
                     const float* weights, float* scores, int num_states) {
    for (int i = 0; i < num_states; i++) {
        scores[i] = (lines[i] * weights[2]) + (heights[i] * weights[0]) + (holes[i] * weights[1]);
    }
}

void evaluate_openmp(const int* heights, const int* holes, const int* lines, 
                     const float* weights, float* scores, int num_states) {
    

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_states; i++) {
        // Setiap thread CPU akan menghitung subset dari 64.000 state
        scores[i] = (lines[i] * weights[2]) + (heights[i] * weights[0]) + (holes[i] * weights[1]);
    }
}