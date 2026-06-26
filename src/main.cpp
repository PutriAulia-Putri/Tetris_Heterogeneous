#include <iostream>
#include <vector>
#include <chrono>
#include <CL/cl.h> 

extern void evaluate_serial(const int*, const int*, const int*, const float*, float*, int);
extern void evaluate_openmp(const int*, const int*, const int*, const float*, float*, int);

int main() {
    const int NUM_STATES = 64000; // Simulasi 64.000 kemungkinan board (Depth-search)
    std::cout << "--- SIMULASI HETEROGENEOUS COMPUTING TETRIS ---\n";
    std::cout << "Mengevaluasi " << NUM_STATES << " posisi balok...\n\n";

    // Host Memory
    std::vector<int> heights(NUM_STATES), holes(NUM_STATES), lines(NUM_STATES);
    std::vector<float> scores_cpu(NUM_STATES, 0), scores_omp(NUM_STATES, 0), scores_gpu(NUM_STATES, 0);
    
    // Bobot heuristik: { Tinggi (-2.0), Lubang (-5.0), Baris (+10.0) }
    float weights[3] = {-2.0f, -5.0f, 10.0f};

    // Mengisi data dummy secara acak untuk simulasi
    for(int i = 0; i < NUM_STATES; i++) {
        heights[i] = rand() % 20;
        holes[i] = rand() % 5;
        lines[i] = rand() % 4;
    }

    // Baseline
    auto start = std::chrono::high_resolution_clock::now();
    evaluate_serial(heights.data(), holes.data(), lines.data(), weights, scores_cpu.data(), NUM_STATES);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time_serial = end - start;
    std::cout << "[1] Waktu Eksekusi Serial CPU : " << time_serial.count() << " ms\n";


    // OPENMP 
    start = std::chrono::high_resolution_clock::now();
    evaluate_openmp(heights.data(), holes.data(), lines.data(), weights, scores_omp.data(), NUM_STATES);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time_omp = end - start;
    std::cout << "[2] Waktu Eksekusi OpenMP     : " << time_omp.count() << " ms (Speedup: " 
              << time_serial.count() / time_omp.count() << "x)\n";

    // OPENCL
    
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL); // Pilih GPU
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    // Membaca file tetris.cl 
    const char* kernel_source = 
        "__kernel void evaluate_moves_kernel(__global const int* h, __global const int* hl, __global const int* lc, __constant float* w, __global float* s, const int num) {"
        "int id = get_global_id(0); if(id < num) { s[id] = (lc[id] * w[2]) + (h[id] * w[0]) + (hl[id] * w[1]); } }";

    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "evaluate_moves_kernel", NULL);

    // Membuat Buffer di Global Memory GPU
    cl_mem d_heights = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * NUM_STATES, heights.data(), NULL);
    cl_mem d_holes = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * NUM_STATES, holes.data(), NULL);
    cl_mem d_lines = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * NUM_STATES, lines.data(), NULL);
    cl_mem d_weights = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * 3, weights, NULL);
    cl_mem d_scores = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * NUM_STATES, NULL, NULL);

    // Set Argumen Kernel
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_heights);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_holes);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_lines);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_weights);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_scores);
    clSetKernelArg(kernel, 5, sizeof(int), &NUM_STATES);

    // Keputusan Desain
    size_t global_work_size = NUM_STATES; 
    
    start = std::chrono::high_resolution_clock::now();
    
    // Eksekusi Kernel GPU
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    clFinish(queue); 
    
    clEnqueueReadBuffer(queue, d_scores, CL_TRUE, 0, sizeof(float) * NUM_STATES, scores_gpu.data(), 0, NULL, NULL);
    
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time_ocl = end - start;
    
    std::cout << "[3] Waktu Eksekusi OpenCL GPU : " << time_ocl.count() << " ms (Speedup: " 
              << time_serial.count() / time_ocl.count() << "x)\n\n";

    std::cout << "Simulasi Selesai. Hasil Speedup terukur dengan baik.\n";

    clReleaseMemObject(d_heights); clReleaseMemObject(d_holes); clReleaseMemObject(d_lines);
    clReleaseMemObject(d_weights); clReleaseMemObject(d_scores);
    clReleaseKernel(kernel); clReleaseProgram(program);
    clReleaseCommandQueue(queue); clReleaseContext(context);

    return 0;
}