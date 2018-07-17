#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

template <class T>
class Vector_h;

template <class T>
class Vector_d;

// host vector
template <class T>
class Vector_h: public thrust::host_vector<T> {
  public:

  // Constructors
  Vector_h() {}
  inline Vector_h(int N) : thrust::host_vector<T>(N) {}
  inline Vector_h(int N, T v) : thrust::host_vector<T>(N,v) {}
  inline Vector_h(const Vector_h<T>& a) : thrust::host_vector<T>(a) {}
  inline Vector_h(const Vector_d<T>& a) : thrust::host_vector<T>(a) {}

  template< typename OtherVector >
    inline void copy( const OtherVector &a ) { 
      this->assign( a.begin( ), a.end( ) ); 
    }

  inline Vector_h<T>& operator=(const Vector_h<T> &a) { copy(a); return *this; }
  inline Vector_h<T>& operator=(const Vector_d<T> &a) { copy(a); return *this; }

  inline T* raw() { 
    if(bytes()>0) return thrust::raw_pointer_cast(this->data()); 
    else return 0;
  } 

  inline const T* raw() const { 
    if(bytes()>0) return thrust::raw_pointer_cast(this->data()); 
    else return 0;
  } 

  inline size_t bytes() const { return this->size()*sizeof(T); }

};

// device vector
template <class T>
class Vector_d: public thrust::device_vector<T> {
  public:

  Vector_d() {}
  inline Vector_d(int N) : thrust::device_vector<T>(N) {}
  inline Vector_d(int N, T v) : thrust::device_vector<T>(N,v) {}
  inline Vector_d(const Vector_d<T>& a) : thrust::device_vector<T>(a) {}
  inline Vector_d(const Vector_h<T>& a) : thrust::device_vector<T>(a) {}

  template< typename OtherVector >
    inline void copy( const OtherVector &a ) { 
      this->assign( a.begin( ), a.end( ) ); 
    }

  inline Vector_d<T>& operator=(const Vector_d<T> &a) { copy(a); return *this; }
  inline Vector_d<T>& operator=(const Vector_h<T> &a) { copy(a); return *this; }

  inline T* raw() { 
    if(bytes()>0) return thrust::raw_pointer_cast(this->data()); 
    else return 0;
  } 

  inline const T* raw() const { 
    if(bytes()>0) return thrust::raw_pointer_cast(this->data()); 
    else return 0;
  } 

  inline size_t bytes() const { return this->size()*sizeof(T); }
};

