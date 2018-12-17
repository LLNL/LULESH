#pragma once

#include <stdio.h>
#include <iostream>
#include <iomanip>

#define cudaSafeCall(x) x;cudaCheckError()

#ifndef NDEBUG

#define cudaCheckError() {                                              \
  cudaDeviceSynchronize();                                              \
  cudaError_t err=cudaGetLastError();                                     \
  if(err!=cudaSuccess) {                                                \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",       \
            __FILE__, __LINE__, cudaGetErrorString( err) );             \
    exit(EXIT_FAILURE);                                                 \
  }                                                                     \
}

#else

#define cudaCheckError() {                                              \
  cudaError_t err=cudaGetLastError();                                     \
  if(err!=cudaSuccess) {                                                \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",       \
            __FILE__, __LINE__, cudaGetErrorString( err) );             \
    exit(EXIT_FAILURE);                                                 \
  }                                                                     \
}

#endif

/* Given a number of bytes, nbytes, and a byte alignment, align, (e.g., 2,
 * 4, 8, or 16), return the smallest integer that is larger than nbytes and
 * a multiple of align.
 */
#define PAD_DIV(nbytes, align)  (((nbytes) + (align) - 1) / (align))
#define PAD(nbytes, align)  (PAD_DIV((nbytes),(align)) * (align))

#if defined(_WIN64) || defined(__LP64__)             
// 64-bit pointer operand constraint for inlined asm
#define _ASM_PTR_ "l"                                        
#else        
// 32-bit pointer operand constraint for inlined asm
#define _ASM_PTR_ "r"
#endif 

// Function to print a vector
template<class vector>
void printVector(const char* label, const vector &v)
{
  std::cout << label << ": ";
  for(int i=0;i<v.size();i++)
  {
    std::cout << std::setprecision(10) << std::setw(14) << v[i] << std::endl;    
  }
  std::cout  << std::endl;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600 
// Double precision atomicAdd in software
static 
__device__ 
__forceinline__
double atomicAdd(double* address, double val)
{
  unsigned long long int* address_as_ull =
    (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = 
      atomicCAS(address_as_ull, assumed,
        __double_as_longlong(val +
          __longlong_as_double(assumed)));
  } while (assumed != old);

  return __longlong_as_double(old);

}
#endif
