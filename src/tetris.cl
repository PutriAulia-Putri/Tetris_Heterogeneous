__kernel void evaluate_moves_kernel(
    __global const int* heights,       
    __global const int* holes,         
    __global const int* lines_cleared, 
    __constant float* weights,         
    __global float* scores,         
    const int num_states) 
{
    // Mendapatkan ID unik dari thread (work-item) yang sedang berjalan
    // Konsep Parallelism: Setiap thread mengurus 1 state papan secara mandiri
    int id = get_global_id(0);

    // Memastikan thread tidak mengakses array di luar batas (Out of Bounds)
    if (id < num_states) {
        // Mengambil data dari Global Memory
        int h = heights[id];
        int hl = holes[id];
        int lc = lines_cleared[id];

        // Membaca bobot dari Constant Memory (w_height, w_holes, w_lines)
        float w_height = weights[0];
        float w_holes  = weights[1];
        float w_lines  = weights[2];

        // Kalkulasi skor Heuristik
        float current_score = (lc * w_lines) + (h * w_height) + (hl * w_holes);

        // Menulis hasil kembali ke Global Memory
        scores[id] = current_score;
    }
}