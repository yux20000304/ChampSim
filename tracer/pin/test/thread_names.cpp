#include <iostream>
#include <omp.h>
#include <vector>
#include <immintrin.h> // For _mm_sfence and _mm_clflush

// Function to issue a magic instruction for ROI markers
void magic_instruction(uint64_t marker)
{
  uint64_t dummyval = 0;
  __asm__ __volatile__("xchg %%rbx, %%rbx;" ::"b"((uint64_t)(&dummyval)), "c"(marker));
}

volatile int data_to_write[4];

void work_function(int region_id)
{
  // Each thread performs some work
  int thread_id = omp_get_thread_num();
  for (int i = 0; i < 10000; i++) {
    // Write data
    data_to_write[thread_id] = i;

    // Persist to memory
    _mm_sfence(); // Prevent store reordering
    _mm_clflush((const void*)&data_to_write[thread_id]); // Flush cache line

#pragma omp critical
    {
      std::cout << "Region " << region_id << ", Thread " << thread_id << ", says hello " << i << std::endl;
    }
  }
}

int main()
{
  // Set the number of threads for the parallel regions
  omp_set_num_threads(4);

  // First parallel region
  std::cout << "Starting first OpenMP parallel region." << std::endl;
#pragma omp parallel
  {
    // All threads start ROI
    magic_instruction(0xEE);

    work_function(1);

    // All threads end ROI
    magic_instruction(0xFF);
  }
  std::cout << "Finished first OpenMP parallel region." << std::endl;

  // Some work in the main thread between parallel regions
  for (int i = 0; i < 5; ++i) {
    std::cout << "Main thread working between regions..." << std::endl;
  }

  // Second parallel region
  std::cout << "Starting second OpenMP parallel region." << std::endl;
#pragma omp parallel
  {
    // All threads start ROI
    magic_instruction(0xEE);

    work_function(2);

    // All threads end ROI
    magic_instruction(0xFF);
  }
  std::cout << "Finished second OpenMP parallel region." << std::endl;

  return 0;
}
