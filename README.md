# Implementasi Heterogeneous Computing pada Simulasi Game Tetris 

Penyusun Proyek: Putri Aulia Rahmah (25032014052)
Mata Kuliah: Arsitektur dan Sistem Komputer 
Tautan Video Presentasi: 

## Deskripsi Proyek
Proyek ini merupakan simulasi evaluasi langkah pada permainan Tetris menggunakan pendekatan Heterogeneous Computing.Tujuan dari projek ini yaitu untuk mengevaluasi 64.000 kemungkinan susunan papan (kondisi depth-search) untuk mencari langkah paling optimal berdasarkan heuristik tertentu (tinggi tumpukan, jumlah lubang, dan baris yang terhapus).

Proyek ini membandingkan tiga metode komputasi:
1. Serial CPU: Eksekusi linear menggunakan 1 core CPU.
2. OpenMP (CPU-Parallelism): Pembagian beban kerja ke seluruh thread CPU menggunakan arahan `#pragma`.
3. OpenCL (GPU-Parallelism): Akselerasi perangkat keras menggunakan eksekusi kernel secara masif (SIMT) pada ribuan core GPU.

## Fitur Sistem
- Pembangkitan data state papan Tetris secara dinamis.
- Perhitungan skor heuristik tersentralisasi.
- Pengukur waktu eksekusi (benchmark timing) beresolusi tinggi (milidetik).
- Perhitungan speedup secara otomatis untuk membandingkan performa paralel vs serial.

## Cara Menjalankan Simulasi
1. Unduh header OpenCL (jika belum ada) menggunakan PowerShell:
2. Lakukan kompilasi program dengan perintah berikut:
   g++ -o tetris.exe main.cpp cpu_omp.cpp -fopenmp -I. C:\Windows\System32\OpenCL.dll
3. Jalankan program simulasi:
   .\tetris.exe