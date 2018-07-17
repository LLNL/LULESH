/*
# Copyright (c) 2011-2012 NVIDIA CORPORATION. All Rights Reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.   
*/

#pragma once

#include <stack>
#include <map>

/***********************************************************
 * Class to allocate arrays of memory for temperary use. 
 * The allocator will hold onto the memory for the next call.
 * This allows memory like Vectors to be reused in different
 * parts of the algorithm without having to store it in 
 * a class and hold onto even when it isn't being used.
 ***********************************************************/
template<typename T>
class Allocator 
{
  typedef std::stack<T*> FreeList;
  typedef std::map<int,FreeList> FreeMap;
    
public:
  static T* allocate( int size );
  static T* free( T *v, int size );
  static T* resize( T *ptr, int old_size, int new_size );
  
  static void clear();

private:
  static FreeMap& getFreeVars( );
    // static FreeMap free_vars;  //a map of vector lists
};
