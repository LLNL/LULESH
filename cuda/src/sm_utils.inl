
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
#define __PTR   "l"
#else
#define __PTR   "r"
#endif

#if __CUDA_ARCH__ >= 700
#define FULL_MASK 0xffffffff
#endif

namespace utils {

// ====================================================================================================================
// Atomics.
// ====================================================================================================================

static __device__ __forceinline__ void atomic_add( float *address, float value )
{
  atomicAdd( address, value );
}

static __device__ __forceinline__ void atomic_add( double *address, double value )
{
  unsigned long long *address_as_ull = (unsigned long long *) address; 
  unsigned long long old = __double_as_longlong( address[0] ), assumed; 
  do { 
    assumed = old; 
    old = atomicCAS( address_as_ull, assumed, __double_as_longlong( value + __longlong_as_double( assumed ) ) ); 
  } 
  while( assumed != old ); 
}

// ====================================================================================================================
// Bit tools.
// ====================================================================================================================

static __device__ __forceinline__ int bfe( int src, int num_bits ) 
{
  unsigned mask;
  asm( "bfe.u32 %0, %1, 0, %2;" : "=r"(mask) : "r"(src), "r"(num_bits) );
  return mask;
}

static __device__ __forceinline__ int bfind( int src ) 
{
  int msb;
  asm( "bfind.u32 %0, %1;" : "=r"(msb) : "r"(src) );
  return msb;
}

static __device__ __forceinline__ int bfind( unsigned long long src ) 
{
  int msb;
  asm( "bfind.u64 %0, %1;" : "=r"(msb) : "l"(src) );
  return msb;
}

static __device__ __forceinline__ unsigned long long brev( unsigned long long src ) 
{
  unsigned long long rev;
  asm( "brev.b64 %0, %1;" : "=l"(rev) : "l"(src) );
  return rev;
}

// ====================================================================================================================
// Warp tools.
// ====================================================================================================================

static __device__ __forceinline__ int lane_id() 
{
  int id;
  asm( "mov.u32 %0, %%laneid;" : "=r"(id) );
  return id;
}

static __device__ __forceinline__ int lane_mask_lt() 
{
  int mask;
  asm( "mov.u32 %0, %%lanemask_lt;" : "=r"(mask) );
  return mask;
}

static __device__ __forceinline__ int warp_id() 
{
  return threadIdx.x >> 5;
}

// ====================================================================================================================
// Loads.
// ====================================================================================================================

enum Ld_mode { LD_AUTO = 0, LD_CA, LD_CG, LD_TEX, LD_NC };

template< Ld_mode Mode >
struct Ld {};

template<>
struct Ld<LD_AUTO> 
{ 
  template< typename T >
  static __device__ __forceinline__ T load( const T *ptr ) { return *ptr; }
};

template<>
struct Ld<LD_CG> 
{ 
  static __device__ __forceinline__ int load( const int *ptr ) 
  { 
    int ret; 
    asm volatile ( "ld.global.cg.s32 %0, [%1];"  : "=r"(ret) : __PTR(ptr) ); 
    return ret; 
  }
  
  static __device__ __forceinline__ float load( const float *ptr ) 
  { 
    float ret; 
    asm volatile ( "ld.global.cg.f32 %0, [%1];"  : "=f"(ret) : __PTR(ptr) ); 
    return ret; 
  }
  
  static __device__ __forceinline__ double load( const double *ptr ) 
  { 
    double ret; 
    asm volatile ( "ld.global.cg.f64 %0, [%1];"  : "=d"(ret) : __PTR(ptr) ); 
    return ret; 
  }
};

template<>
struct Ld<LD_CA> 
{ 
  static __device__ __forceinline__ int load( const int *ptr ) 
  { 
    int ret; 
    asm volatile ( "ld.global.ca.s32 %0, [%1];"  : "=r"(ret) : __PTR(ptr) ); 
    return ret; 
  }
  
  static __device__ __forceinline__ float load( const float *ptr ) 
  { 
    float ret; 
    asm volatile ( "ld.global.ca.f32 %0, [%1];"  : "=f"(ret) : __PTR(ptr) ); 
    return ret; 
  }
  
  static __device__ __forceinline__ double load( const double *ptr ) 
  { 
    double ret; 
    asm volatile ( "ld.global.ca.f64 %0, [%1];"  : "=d"(ret) : __PTR(ptr) ); 
    return ret; 
  }
};

template<>
struct Ld<LD_NC> 
{ 
  template< typename T >
  static __device__ __forceinline__ T load( const T *ptr ) { return __ldg( ptr ); }
};


// ====================================================================================================================
// Vector loads.
// ====================================================================================================================

static __device__ __forceinline__ void load_vec2( float (&u)[2], const float *ptr )
{
  asm( "ld.global.cg.f32.v2 {%0, %1}, [%2];" : "=f"(u[0]), "=f"(u[1]) : __PTR(ptr) );
}

static __device__ __forceinline__ void load_vec2( double (&u)[2], const double *ptr )
{
  asm( "ld.global.cg.f64.v2 {%0, %1}, [%2];" : "=d"(u[0]), "=d"(u[1]) : __PTR(ptr) );
}

static __device__ __forceinline__ void load_vec4( float (&u)[4], const float *ptr )
{
  asm( "ld.global.cg.f32.v4 {%0, %1, %2, %3}, [%4];" : "=f"(u[0]), "=f"(u[1]), "=f"(u[2]), "=f"(u[3]) : __PTR(ptr) );
}

static __device__ __forceinline__ void load_vec4( double (&u)[4], const double *ptr )
{
  asm( "ld.global.cg.f64.v2 {%0, %1}, [%2];" : "=d"(u[0]), "=d"(u[1]) : __PTR(ptr + 0) );
  asm( "ld.global.cg.f64.v2 {%0, %1}, [%2];" : "=d"(u[2]), "=d"(u[3]) : __PTR(ptr + 2) );
}

// ====================================================================================================================
// Shuffle.
// ====================================================================================================================
static __device__ __forceinline__ float shfl( float r, int lane, int warp_size )
{
#if __CUDA_ARCH__ >= 700
  return __shfl_sync(FULL_MASK, r, lane , warp_size);
#elif __CUDA_ARCH__ >= 300
  return __shfl( r, lane , warp_size);
#else
  return 0.0f;
#endif
}

static __device__ __forceinline__ double shfl( double r, int lane, int warp_size )
{
#if __CUDA_ARCH__ >= 700
  int hi = __shfl_sync(FULL_MASK, __double2hiint(r), lane , warp_size);
  int lo = __shfl_sync(FULL_MASK, __double2loint(r), lane , warp_size);
  return __hiloint2double( hi, lo );
#elif __CUDA_ARCH__ >= 300
  int hi = __shfl( __double2hiint(r), lane , warp_size);
  int lo = __shfl( __double2loint(r), lane , warp_size);
  return __hiloint2double( hi, lo );
#else
  return 0.0;
#endif
}

static __device__ __forceinline__ float shfl_xor( float r, int mask, int warp_size )
{
#if __CUDA_ARCH__ >= 700
  return __shfl_xor_sync(FULL_MASK, r, mask, warp_size );
#elif __CUDA_ARCH__ >= 300
  return __shfl_xor( r, mask, warp_size );
#else
  return 0.0f;
#endif
}

static __device__ __forceinline__ double shfl_xor( double r, int mask, int warp_size )
{
#if __CUDA_ARCH__ >= 700
  int hi = __shfl_xor_sync( __double2hiint(r), mask, warp_size );
  int lo = __shfl_xor_sync( __double2loint(r), mask, warp_size );
  return __hiloint2double( hi, lo );
#elif __CUDA_ARCH__ >= 300
  int hi = __shfl_xor( __double2hiint(r), mask, warp_size );
  int lo = __shfl_xor( __double2loint(r), mask, warp_size );
  return __hiloint2double( hi, lo );
#else
  return 0.0;
#endif
}

static __device__ __forceinline__ float shfl_down( float r, int offset )
{
#if __CUDA_ARCH__ >= 700
  return __shfl_down_sync(FULL_MASK, r, offset );
#elif __CUDA_ARCH__ >= 300
  return __shfl_down( r, offset );
#else
  return 0.0f;
#endif
}

static __device__ __forceinline__ double shfl_down( double r, int offset )
{
#if __CUDA_ARCH__ >= 700
  int hi = __shfl_down_sync(FULL_MASK, __double2hiint(r), offset );
  int lo = __shfl_down_sync(FULL_MASK, __double2loint(r), offset );
  return __hiloint2double( hi, lo );
#elif __CUDA_ARCH__ >= 300
  int hi = __shfl_down( __double2hiint(r), offset );
  int lo = __shfl_down( __double2loint(r), offset );
  return __hiloint2double( hi, lo );
#else
  return 0.0;
#endif
}

// ====================================================================================================================
// Warp-level reductions.
// ====================================================================================================================

struct Add
{
  template< typename Value_type >
  static __device__ __forceinline__ Value_type eval( Value_type x, Value_type y ) { return x+y; }
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300

template< int NUM_THREADS_PER_ITEM, int WARP_SIZE >
struct Warp_reduce_pow2
{
  template< typename Operator, typename Value_type >
  static __device__ __inline__ Value_type execute( Value_type x )
  {
    #pragma unroll
    for( int mask = WARP_SIZE / 2 ; mask >= NUM_THREADS_PER_ITEM ; mask >>= 1 )
      x = Operator::eval( x, shfl_xor(x, mask) );
    return x;
  }
};

template< int NUM_THREADS_PER_ITEM, int WARP_SIZE >
struct Warp_reduce_linear
{
  template< typename Operator, typename Value_type >
  static __device__ __inline__ Value_type execute( Value_type x )
  {
    const int NUM_STEPS = WARP_SIZE / NUM_THREADS_PER_ITEM;
    int my_lane_id = utils::lane_id();
    #pragma unroll
    for( int i = 1 ; i < NUM_STEPS ; ++i )
    {
      Value_type y = shfl_down( x, i*NUM_THREADS_PER_ITEM );
      if( my_lane_id < NUM_THREADS_PER_ITEM )
        x = Operator::eval( x, y );
    }
    return x;
  }
};

#else

template< int NUM_THREADS_PER_ITEM, int WARP_SIZE >
struct Warp_reduce_pow2
{
  template< typename Operator, typename Value_type >
  static __device__ __inline__ Value_type execute( volatile Value_type *smem, Value_type x )
  {
    int my_lane_id = utils::lane_id();
    #pragma unroll
    for( int offset = WARP_SIZE / 2 ; offset >= NUM_THREADS_PER_ITEM ; offset >>= 1 )
      if( my_lane_id < offset )
        smem[threadIdx.x] = x = Operator::eval( x, smem[threadIdx.x+offset] );
    return x;
  }
};

template< int NUM_THREADS_PER_ITEM, int WARP_SIZE >
struct Warp_reduce_linear
{
  template< typename Operator, typename Value_type >
  static __device__ __inline__ Value_type execute( volatile Value_type *smem, Value_type x )
  {
    const int NUM_STEPS = WARP_SIZE / NUM_THREADS_PER_ITEM;
    int my_lane_id = utils::lane_id();
    #pragma unroll
    for( int i = 1 ; i < NUM_STEPS ; ++i )
      if( my_lane_id < NUM_THREADS_PER_ITEM )
        smem[threadIdx.x] = x = Operator::eval( x, smem[threadIdx.x+i*NUM_THREADS_PER_ITEM] );
    return x;
  }
};

#endif

// ====================================================================================================================

template< int NUM_THREADS_PER_ITEM, int WARP_SIZE = 32 >
struct Warp_reduce : public Warp_reduce_pow2<NUM_THREADS_PER_ITEM, WARP_SIZE> {};

template< int WARP_SIZE >
struct Warp_reduce< 3, WARP_SIZE> : public Warp_reduce_linear< 3, WARP_SIZE> {};

template< int WARP_SIZE >
struct Warp_reduce< 4, WARP_SIZE> : public Warp_reduce_linear< 4, WARP_SIZE> {};

template< int WARP_SIZE >
struct Warp_reduce< 5, WARP_SIZE> : public Warp_reduce_linear< 5, WARP_SIZE> {};

template< int WARP_SIZE >
struct Warp_reduce< 6, WARP_SIZE> : public Warp_reduce_linear< 6, WARP_SIZE> {};

template< int WARP_SIZE >
struct Warp_reduce< 7, WARP_SIZE> : public Warp_reduce_linear< 7, WARP_SIZE> {};

template< int WARP_SIZE >
struct Warp_reduce< 9, WARP_SIZE> : public Warp_reduce_linear< 9, WARP_SIZE> {};

template< int WARP_SIZE >
struct Warp_reduce<10, WARP_SIZE> : public Warp_reduce_linear<10, WARP_SIZE> {};

template< int WARP_SIZE >
struct Warp_reduce<11, WARP_SIZE> : public Warp_reduce_linear<11, WARP_SIZE> {};

template< int WARP_SIZE >
struct Warp_reduce<12, WARP_SIZE> : public Warp_reduce_linear<12, WARP_SIZE> {};

template< int WARP_SIZE >
struct Warp_reduce<13, WARP_SIZE> : public Warp_reduce_linear<13, WARP_SIZE> {};

template< int WARP_SIZE >
struct Warp_reduce<14, WARP_SIZE> : public Warp_reduce_linear<14, WARP_SIZE> {};

template< int WARP_SIZE >
struct Warp_reduce<15, WARP_SIZE> : public Warp_reduce_linear<15, WARP_SIZE> {};

// ====================================================================================================================

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300

template< int NUM_THREADS_PER_ITEM, typename Operator, typename Value_type >
static __device__ __forceinline__ Value_type warp_reduce( Value_type x )
{
  return Warp_reduce<NUM_THREADS_PER_ITEM>::template execute<Operator>( x );
}

#else

template< int NUM_THREADS_PER_ITEM, typename Operator, typename Value_type >
static __device__ __forceinline__ Value_type warp_reduce( volatile Value_type *smem, Value_type x )
{
  return Warp_reduce<NUM_THREADS_PER_ITEM>::template execute<Operator>( smem, x );
}

#endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace utils

