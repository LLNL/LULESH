/*
# Copyright (c) 2011-2012 NVIDIA CORPORATION. All Rights Reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.   
*/

#include <allocator.h>
#include <stack>
#include <map>
#include <vector.h>

template< typename T >
inline T* allocate( int size ) 
{
  return new T( size );
}

template< typename T >
Allocator<T>::FreeMap& Allocator<T>::getFreeVars( )
{
  static FreeMap s_free_vars;
  return s_free_vars;
}

template<typename T>
T* 
Allocator<T>::allocate(int size) 
{
	T *v;
	bool is_empty;
    {
		//locate free var list for the right size
		FreeList &f_vars=getFreeVars( )[size];
		is_empty = f_vars.empty();
		if (!is_empty) {
			//set the return value to the previously freed vector
			v=f_vars.top();
			//remove the vector from the free vector list
			f_vars.pop();
		}

	}
	if(is_empty) //if there are no free vectors
	{
		//create a new vector
		//v=allocate<T>(size);
    v = new T(size);

	}
	return v;
}

template<typename T>
T* Allocator<T>::free(T* v,int size) {
	//add the vector to the free vector list
	{
    if( v != NULL )
		getFreeVars( )[size].push(v);
	}
  return NULL;
}

template<typename T>
T* 
Allocator<T>::resize( T *ptr, int old_size, int new_size ) 
{
  if( old_size == new_size )
    return ptr;
  free( ptr, old_size );
  return allocate( new_size );
}

template<typename T>
void Allocator<T>::clear() {
	{
		FreeMap &free_vars = getFreeVars( );
		for(typename FreeMap::iterator m_iter=free_vars.begin();m_iter!=free_vars.end();m_iter++)
		{
			FreeList &stack=m_iter->second;
			while(!stack.empty()) {
				T *ptr = stack.top();
				stack.pop();
				delete ptr;
			}
		}
		free_vars.clear();
	}
}

/****************************************
 * Explict instantiations
 ***************************************/

// here we imply Index_Type is int
template class Allocator<Vector_d<float> >;
template class Allocator<Vector_d<double> >;
template class Allocator<Vector_d<int> >;

template class Allocator<Vector_h<float> >;
template class Allocator<Vector_h<double> >;
template class Allocator<Vector_h<int> >;
