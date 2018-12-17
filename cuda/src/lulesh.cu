/*

                 Copyright (c) 2010.
      Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory.
                  LLNL-CODE-461231
                All rights reserved.

This file is part of LULESH, Version 1.0.
Please also read this link -- http://www.opensource.org/licenses/index.php

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the disclaimer below.

   * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the disclaimer (as noted below)
     in the documentation and/or other materials provided with the
     distribution.

   * Neither the name of the LLNS/LLNL nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC,
THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Additional BSD Notice

1. This notice is required to be provided under our contract with the U.S.
   Department of Energy (DOE). This work was produced at Lawrence Livermore
   National Laboratory under Contract No. DE-AC52-07NA27344 with the DOE.

2. Neither the United States Government nor Lawrence Livermore National
   Security, LLC nor any of their employees, makes any warranty, express
   or implied, or assumes any liability or responsibility for the accuracy,
   completeness, or usefulness of any information, apparatus, product, or
   process disclosed, or represents that its use would not infringe
   privately-owned rights.

3. Also, reference herein to any specific commercial products, process, or
   services by trade name, trademark, manufacturer or otherwise does not
   necessarily constitute or imply its endorsement, recommendation, or
   favoring by the United States Government or Lawrence Livermore National
   Security, LLC. The views and opinions of authors expressed herein do not
   necessarily state or reflect those of the United States Government or
   Lawrence Livermore National Security, LLC, and shall not be used for
   advertising or product endorsement purposes.

*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <sstream>

#include <util.h>
#include <sm_utils.inl>
#include <cuda.h>
#include <allocator.h>
#include "cuda_profiler_api.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <sys/time.h>
#include <unistd.h>

#include "lulesh.h"

/****************************************************/
/* Allow flexibility for arithmetic representations */
/****************************************************/
__device__ inline real4  SQRT(real4  arg) { return sqrtf(arg) ; }
__device__ inline real8  SQRT(real8  arg) { return sqrt(arg) ; }

__device__ inline real4  CBRT(real4  arg) { return cbrtf(arg) ; }
__device__ inline real8  CBRT(real8  arg) { return cbrt(arg) ; }

__device__ __host__ inline real4  FABS(real4  arg) { return fabsf(arg) ; }
__device__ __host__ inline real8  FABS(real8  arg) { return fabs(arg) ; }

__device__ inline real4  FMAX(real4  arg1,real4  arg2) { return fmaxf(arg1,arg2) ; }
__device__ inline real8  FMAX(real8  arg1,real8  arg2) { return fmax(arg1,arg2) ; }

#define MAX(a, b) ( ((a) > (b)) ? (a) : (b))

/* Stuff needed for boundary conditions */
/* 2 BCs on each of 6 hexahedral faces (12 bits) */
#define XI_M        0x00007
#define XI_M_SYMM   0x00001
#define XI_M_FREE   0x00002
#define XI_M_COMM   0x00004

#define XI_P        0x00038
#define XI_P_SYMM   0x00008
#define XI_P_FREE   0x00010
#define XI_P_COMM   0x00020

#define ETA_M       0x001c0
#define ETA_M_SYMM  0x00040
#define ETA_M_FREE  0x00080
#define ETA_M_COMM  0x00100

#define ETA_P       0x00e00
#define ETA_P_SYMM  0x00200
#define ETA_P_FREE  0x00400
#define ETA_P_COMM  0x00800

#define ZETA_M      0x07000
#define ZETA_M_SYMM 0x01000
#define ZETA_M_FREE 0x02000
#define ZETA_M_COMM 0x04000

#define ZETA_P      0x38000
#define ZETA_P_SYMM 0x08000
#define ZETA_P_FREE 0x10000
#define ZETA_P_COMM 0x20000

#define VOLUDER(a0,a1,a2,a3,a4,a5,b0,b1,b2,b3,b4,b5,dvdc)		\
{									\
  const Real_t twelfth = Real_t(1.0) / Real_t(12.0) ;			\
									\
   dvdc= 								\
     ((a1) + (a2)) * ((b0) + (b1)) - ((a0) + (a1)) * ((b1) + (b2)) +	\
     ((a0) + (a4)) * ((b3) + (b4)) - ((a3) + (a4)) * ((b0) + (b4)) -	\
     ((a2) + (a5)) * ((b3) + (b5)) + ((a3) + (a5)) * ((b2) + (b5));	\
   dvdc *= twelfth;							\
}
/*
__device__
static 
__forceinline__
void SumOverNodes(Real_t& val, volatile Real_t* smem, int cta_elem, int node) {

  int tid = (cta_elem << 3) + node;
  smem[tid] = val;
  if (node < 4)
  {
    smem[tid] += smem[tid+4];
    smem[tid] += smem[tid+2];
    smem[tid] += smem[tid+1];
  }
  val = smem[(cta_elem << 3)];
}
*/

__device__
static 
__forceinline__
void SumOverNodesShfl(Real_t& val) {

   val += utils::shfl_xor( val, 4, 8);
   val += utils::shfl_xor( val, 2, 8);
   val += utils::shfl_xor( val, 1, 8);
}

__host__ __device__
static 
__forceinline__ 
Real_t CalcElemVolume( const Real_t x0, const Real_t x1,
               const Real_t x2, const Real_t x3,
               const Real_t x4, const Real_t x5,
               const Real_t x6, const Real_t x7,
               const Real_t y0, const Real_t y1,
               const Real_t y2, const Real_t y3,
               const Real_t y4, const Real_t y5,
               const Real_t y6, const Real_t y7,
               const Real_t z0, const Real_t z1,
               const Real_t z2, const Real_t z3,
               const Real_t z4, const Real_t z5,
               const Real_t z6, const Real_t z7 )
{
  Real_t twelveth = Real_t(1.0)/Real_t(12.0);

  Real_t dx61 = x6 - x1;
  Real_t dy61 = y6 - y1;
  Real_t dz61 = z6 - z1;

  Real_t dx70 = x7 - x0;
  Real_t dy70 = y7 - y0;
  Real_t dz70 = z7 - z0;

  Real_t dx63 = x6 - x3;
  Real_t dy63 = y6 - y3;
  Real_t dz63 = z6 - z3;

  Real_t dx20 = x2 - x0;
  Real_t dy20 = y2 - y0;
  Real_t dz20 = z2 - z0;

  Real_t dx50 = x5 - x0;
  Real_t dy50 = y5 - y0;
  Real_t dz50 = z5 - z0;

  Real_t dx64 = x6 - x4;
  Real_t dy64 = y6 - y4;
  Real_t dz64 = z6 - z4;

  Real_t dx31 = x3 - x1;
  Real_t dy31 = y3 - y1;
  Real_t dz31 = z3 - z1;

  Real_t dx72 = x7 - x2;
  Real_t dy72 = y7 - y2;
  Real_t dz72 = z7 - z2;

  Real_t dx43 = x4 - x3;
  Real_t dy43 = y4 - y3;
  Real_t dz43 = z4 - z3;

  Real_t dx57 = x5 - x7;
  Real_t dy57 = y5 - y7;
  Real_t dz57 = z5 - z7;

  Real_t dx14 = x1 - x4;
  Real_t dy14 = y1 - y4;
  Real_t dz14 = z1 - z4;

  Real_t dx25 = x2 - x5;
  Real_t dy25 = y2 - y5;
  Real_t dz25 = z2 - z5;

#define TRIPLE_PRODUCT(x1, y1, z1, x2, y2, z2, x3, y3, z3) \
   ((x1)*((y2)*(z3) - (z2)*(y3)) + (x2)*((z1)*(y3) - (y1)*(z3)) + (x3)*((y1)*(z2) - (z1)*(y2)))

  // 11 + 3*14
  Real_t volume =
    TRIPLE_PRODUCT(dx31 + dx72, dx63, dx20,
       dy31 + dy72, dy63, dy20,
       dz31 + dz72, dz63, dz20) +
    TRIPLE_PRODUCT(dx43 + dx57, dx64, dx70,
       dy43 + dy57, dy64, dy70,
       dz43 + dz57, dz64, dz70) +
    TRIPLE_PRODUCT(dx14 + dx25, dx61, dx50,
       dy14 + dy25, dy61, dy50,
       dz14 + dz25, dz61, dz50);

#undef TRIPLE_PRODUCT

  volume *= twelveth;

  return volume ;
}

__host__ __device__
static 
__forceinline__
Real_t CalcElemVolume( const Real_t x[8], const Real_t y[8], const Real_t z[8] )
{
return CalcElemVolume( x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],
                       y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7],
                       z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7]);
}

void cuda_init(int rank)
{
    Int_t deviceCount, dev;
    cudaDeviceProp cuda_deviceProp;
    
    cudaSafeCall( cudaGetDeviceCount(&deviceCount) );
    if (deviceCount == 0) {
        fprintf(stderr, "cuda_init(): no devices supporting CUDA.\n");
        exit(1);
    }

    dev = rank % deviceCount;

    if ((dev < 0) || (dev > deviceCount-1)) {
        fprintf(stderr, "cuda_init(): requested device (%d) out of range [%d,%d]\n",
                dev, 0, deviceCount-1);
        exit(1);
    }

    cudaSafeCall( cudaSetDevice(dev) );

    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, dev);

    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    printf("Host %s using GPU %i: %s\n", hostname, dev, props.name);

    cudaSafeCall( cudaGetDeviceProperties(&cuda_deviceProp, dev) );
    if (cuda_deviceProp.major < 3) {
        fprintf(stderr, "cuda_init(): This implementation of Lulesh requires device SM 3.0+.\n", dev);
        exit(1);
    }

#if CUDART_VERSION < 5000
   fprintf(stderr,"cuda_init(): This implementation of Lulesh uses texture objects, which is requires Cuda 5.0+.\n");
   exit(1);
#endif

}

void AllocateNodalPersistent(Domain* domain, size_t domNodes)
{
  domain->x.resize(domNodes) ;  /* coordinates */
  domain->y.resize(domNodes) ;
  domain->z.resize(domNodes) ;

  domain->xd.resize(domNodes) ; /* velocities */
  domain->yd.resize(domNodes) ;
  domain->zd.resize(domNodes) ;

  domain->xdd.resize(domNodes) ; /* accelerations */
  domain->ydd.resize(domNodes) ;
  domain->zdd.resize(domNodes) ;

  domain->fx.resize(domNodes) ;  /* forces */
  domain->fy.resize(domNodes) ;
  domain->fz.resize(domNodes) ;

  domain->nodalMass.resize(domNodes) ;  /* mass */
}

void AllocateElemPersistent(Domain* domain, size_t domElems, size_t padded_domElems)
{
   domain->matElemlist.resize(domElems) ;  /* material indexset */
   domain->nodelist.resize(8*padded_domElems) ;   /* elemToNode connectivity */

   domain->lxim.resize(domElems) ; /* elem connectivity through face */
   domain->lxip.resize(domElems) ;
   domain->letam.resize(domElems) ;
   domain->letap.resize(domElems) ;
   domain->lzetam.resize(domElems) ;
   domain->lzetap.resize(domElems) ;

   domain->elemBC.resize(domElems) ;  /* elem face symm/free-surf flag */

   domain->e.resize(domElems) ;   /* energy */
   domain->p.resize(domElems) ;   /* pressure */

   domain->q.resize(domElems) ;   /* q */
   domain->ql.resize(domElems) ;  /* linear term for q */
   domain->qq.resize(domElems) ;  /* quadratic term for q */

   domain->v.resize(domElems) ;     /* relative volume */

   domain->volo.resize(domElems) ;  /* reference volume */
   domain->delv.resize(domElems) ;  /* m_vnew - m_v */
   domain->vdov.resize(domElems) ;  /* volume derivative over volume */

   domain->arealg.resize(domElems) ;  /* elem characteristic length */

   domain->ss.resize(domElems) ;      /* "sound speed" */

   domain->elemMass.resize(domElems) ;  /* mass */

}

void AllocateSymmX(Domain* domain, size_t size)
{
   domain->symmX.resize(size) ;
}

void AllocateSymmY(Domain* domain, size_t size)
{
   domain->symmY.resize(size) ;
}

void AllocateSymmZ(Domain* domain, size_t size)
{
   domain->symmZ.resize(size) ;
}

void InitializeFields(Domain* domain)
{
 /* Basic Field Initialization */

 thrust::fill(domain->ss.begin(),domain->ss.end(),0.);
 thrust::fill(domain->e.begin(),domain->e.end(),0.);
 thrust::fill(domain->p.begin(),domain->p.end(),0.);
 thrust::fill(domain->q.begin(),domain->q.end(),0.);
 thrust::fill(domain->v.begin(),domain->v.end(),1.);

 thrust::fill(domain->xd.begin(),domain->xd.end(),0.);
 thrust::fill(domain->yd.begin(),domain->yd.end(),0.);
 thrust::fill(domain->zd.begin(),domain->zd.end(),0.);

 thrust::fill(domain->xdd.begin(),domain->xdd.end(),0.);
 thrust::fill(domain->ydd.begin(),domain->ydd.end(),0.);
 thrust::fill(domain->zdd.begin(),domain->zdd.end(),0.);

 thrust::fill(domain->nodalMass.begin(),domain->nodalMass.end(),0.);
}

////////////////////////////////////////////////////////////////////////////////
void
Domain::SetupCommBuffers(Int_t edgeNodes)
{
  // allocate a buffer large enough for nodal ghost data 
  maxEdgeSize = MAX(this->sizeX, MAX(this->sizeY, this->sizeZ))+1 ;
  maxPlaneSize = CACHE_ALIGN_REAL(maxEdgeSize*maxEdgeSize) ;
  maxEdgeSize = CACHE_ALIGN_REAL(maxEdgeSize) ;

  // assume communication to 6 neighbors by default 
  m_rowMin = (m_rowLoc == 0)        ? 0 : 1;
  m_rowMax = (m_rowLoc == m_tp-1)     ? 0 : 1;
  m_colMin = (m_colLoc == 0)        ? 0 : 1;
  m_colMax = (m_colLoc == m_tp-1)     ? 0 : 1;
  m_planeMin = (m_planeLoc == 0)    ? 0 : 1;
  m_planeMax = (m_planeLoc == m_tp-1) ? 0 : 1;

#if USE_MPI   
  // account for face communication 
  Index_t comBufSize =
    (m_rowMin + m_rowMax + m_colMin + m_colMax + m_planeMin + m_planeMax) *
    maxPlaneSize * MAX_FIELDS_PER_MPI_COMM ;

  // account for edge communication 
  comBufSize +=
    ((m_rowMin & m_colMin) + (m_rowMin & m_planeMin) + (m_colMin & m_planeMin) +
     (m_rowMax & m_colMax) + (m_rowMax & m_planeMax) + (m_colMax & m_planeMax) +
     (m_rowMax & m_colMin) + (m_rowMin & m_planeMax) + (m_colMin & m_planeMax) +
     (m_rowMin & m_colMax) + (m_rowMax & m_planeMin) + (m_colMax & m_planeMin)) *
    maxPlaneSize * MAX_FIELDS_PER_MPI_COMM ;

  // account for corner communication 
  // factor of 16 is so each buffer has its own cache line 
  comBufSize += ((m_rowMin & m_colMin & m_planeMin) +
                 (m_rowMin & m_colMin & m_planeMax) +
                 (m_rowMin & m_colMax & m_planeMin) +
                 (m_rowMin & m_colMax & m_planeMax) +
                 (m_rowMax & m_colMin & m_planeMin) +
                 (m_rowMax & m_colMin & m_planeMax) +
                 (m_rowMax & m_colMax & m_planeMin) +
                 (m_rowMax & m_colMax & m_planeMax)) * CACHE_COHERENCE_PAD_REAL ;

  this->commDataSend = new Real_t[comBufSize] ;
  this->commDataRecv = new Real_t[comBufSize] ;

  // pin buffers
  cudaHostRegister(this->commDataSend, comBufSize*sizeof(Real_t), 0);
  cudaHostRegister(this->commDataRecv, comBufSize*sizeof(Real_t), 0);

  // prevent floating point exceptions 
  memset(this->commDataSend, 0, comBufSize*sizeof(Real_t)) ;
  memset(this->commDataRecv, 0, comBufSize*sizeof(Real_t)) ;

  // allocate shadow GPU buffers
  cudaMalloc(&this->d_commDataSend, comBufSize*sizeof(Real_t));
  cudaMalloc(&this->d_commDataRecv, comBufSize*sizeof(Real_t));
  
  // prevent floating point exceptions 
  cudaMemset(this->d_commDataSend, 0, comBufSize*sizeof(Real_t));
  cudaMemset(this->d_commDataRecv, 0, comBufSize*sizeof(Real_t));
#endif
}

void SetupConnectivityBC(Domain *domain, int edgeElems)
{
  int domElems = domain->numElem;

  Vector_h<Index_t> lxim_h(domElems);
  Vector_h<Index_t> lxip_h(domElems);
  Vector_h<Index_t> letam_h(domElems);
  Vector_h<Index_t> letap_h(domElems);
  Vector_h<Index_t> lzetam_h(domElems);
  Vector_h<Index_t> lzetap_h(domElems);

    /* set up elemement connectivity information */
    lxim_h[0] = 0 ;
    for (Index_t i=1; i<domElems; ++i) {
       lxim_h[i]   = i-1 ;
       lxip_h[i-1] = i ;
    }
    lxip_h[domElems-1] = domElems-1 ;

    for (Index_t i=0; i<edgeElems; ++i) {
       letam_h[i] = i ;
       letap_h[domElems-edgeElems+i] = domElems-edgeElems+i ;
    }
    for (Index_t i=edgeElems; i<domElems; ++i) {
       letam_h[i] = i-edgeElems ;
       letap_h[i-edgeElems] = i ;
    }

    for (Index_t i=0; i<edgeElems*edgeElems; ++i) {
       lzetam_h[i] = i ;
       lzetap_h[domElems-edgeElems*edgeElems+i] = domElems-edgeElems*edgeElems+i ;
    }
    for (Index_t i=edgeElems*edgeElems; i<domElems; ++i) {
       lzetam_h[i] = i - edgeElems*edgeElems ;
       lzetap_h[i-edgeElems*edgeElems] = i ;
    }


  /* set up boundary condition information */
  Vector_h<Index_t> elemBC_h(domElems);
  for (Index_t i=0; i<domElems; ++i) {
     elemBC_h[i] = 0 ;  /* clear BCs by default */
  }

  Index_t ghostIdx[6] ;  // offsets to ghost locations

  for (Index_t i=0; i<6; ++i) {
    ghostIdx[i] = INT_MIN ;
  }

  Int_t pidx = domElems ;
  if (domain->m_planeMin != 0) {
    ghostIdx[0] = pidx ;
    pidx += domain->sizeX*domain->sizeY ;
  }

  if (domain->m_planeMax != 0) {
    ghostIdx[1] = pidx ;
    pidx += domain->sizeX*domain->sizeY ;
  }

  if (domain->m_rowMin != 0) {
    ghostIdx[2] = pidx ;
    pidx += domain->sizeX*domain->sizeZ ;
  }

  if (domain->m_rowMax != 0) {
    ghostIdx[3] = pidx ;
    pidx += domain->sizeX*domain->sizeZ ;
  }

  if (domain->m_colMin != 0) {
    ghostIdx[4] = pidx ;
    pidx += domain->sizeY*domain->sizeZ ;
  }

  if (domain->m_colMax != 0) {
    ghostIdx[5] = pidx ;
  }

  /* symmetry plane or free surface BCs */
  for (Index_t i=0; i<edgeElems; ++i) {
    Index_t planeInc = i*edgeElems*edgeElems ;
    Index_t rowInc   = i*edgeElems ;
    for (Index_t j=0; j<edgeElems; ++j) {
      if (domain->m_planeLoc == 0) {
        elemBC_h[rowInc+j] |= ZETA_M_SYMM ;
      }
      else {
        elemBC_h[rowInc+j] |= ZETA_M_COMM ;
        lzetam_h[rowInc+j] = ghostIdx[0] + rowInc + j ;
      }

      if (domain->m_planeLoc == domain->m_tp-1) {
        elemBC_h[rowInc+j+domElems-edgeElems*edgeElems] |=
          ZETA_P_FREE;
      }
      else {
        elemBC_h[rowInc+j+domElems-edgeElems*edgeElems] |=
          ZETA_P_COMM ;
        lzetap_h[rowInc+j+domElems-edgeElems*edgeElems] =
          ghostIdx[1] + rowInc + j ;
      }

      if (domain->m_rowLoc == 0) {
        elemBC_h[planeInc+j] |= ETA_M_SYMM ;
      }
      else {
        elemBC_h[planeInc+j] |= ETA_M_COMM ;
        letam_h[planeInc+j] = ghostIdx[2] + rowInc + j ;
      }

      if (domain->m_rowLoc == domain->m_tp-1) {
        elemBC_h[planeInc+j+edgeElems*edgeElems-edgeElems] |=
          ETA_P_FREE ;
      }
      else {
        elemBC_h[planeInc+j+edgeElems*edgeElems-edgeElems] |=
          ETA_P_COMM ;
        letap_h[planeInc+j+edgeElems*edgeElems-edgeElems] =
          ghostIdx[3] +  rowInc + j ;
      }

      if (domain->m_colLoc == 0) {
        elemBC_h[planeInc+j*edgeElems] |= XI_M_SYMM ;
      }
      else {
        elemBC_h[planeInc+j*edgeElems] |= XI_M_COMM ;
        lxim_h[planeInc+j*edgeElems] = ghostIdx[4] + rowInc + j ;
      }

      if (domain->m_colLoc == domain->m_tp-1) {
        elemBC_h[planeInc+j*edgeElems+edgeElems-1] |= XI_P_FREE ;
      }
      else {
        elemBC_h[planeInc+j*edgeElems+edgeElems-1] |= XI_P_COMM ;
        lxip_h[planeInc+j*edgeElems+edgeElems-1] =
          ghostIdx[5] + rowInc + j ;
      }
    }
  }

  domain->elemBC = elemBC_h;
  domain->lxim = lxim_h;
  domain->lxip = lxip_h;
  domain->letam = letam_h;
  domain->letap = letap_h;
  domain->lzetam = lzetam_h;
  domain->lzetap = lzetap_h;
}

void Domain::BuildMesh(Int_t nx, Int_t edgeNodes, Int_t edgeElems, Int_t domNodes, Int_t padded_domElems, Vector_h<Real_t> &x_h, Vector_h<Real_t> &y_h, Vector_h<Real_t> &z_h, Vector_h<Int_t> &nodelist_h)
{
  Index_t meshEdgeElems = m_tp*nx ;

  x_h.resize(domNodes);
  y_h.resize(domNodes);
  z_h.resize(domNodes);

  // initialize nodal coordinates 
  Index_t nidx = 0 ;
  Real_t tz = Real_t(1.125)*Real_t(m_planeLoc*nx)/Real_t(meshEdgeElems) ;
  for (Index_t plane=0; plane<edgeNodes; ++plane) {
    Real_t ty = Real_t(1.125)*Real_t(m_rowLoc*nx)/Real_t(meshEdgeElems) ;
    for (Index_t row=0; row<edgeNodes; ++row) {
      Real_t tx = Real_t(1.125)*Real_t(m_colLoc*nx)/Real_t(meshEdgeElems) ;
      for (Index_t col=0; col<edgeNodes; ++col) {
        x_h[nidx] = tx ;
        y_h[nidx] = ty ;
        z_h[nidx] = tz ;
        ++nidx ;
        // tx += ds ; // may accumulate roundoff... 
        tx = Real_t(1.125)*Real_t(m_colLoc*nx+col+1)/Real_t(meshEdgeElems) ;
      }
      // ty += ds ;  // may accumulate roundoff... 
      ty = Real_t(1.125)*Real_t(m_rowLoc*nx+row+1)/Real_t(meshEdgeElems) ;
    }
    // tz += ds ;  // may accumulate roundoff... 
    tz = Real_t(1.125)*Real_t(m_planeLoc*nx+plane+1)/Real_t(meshEdgeElems) ;
  }

  x = x_h;
  y = y_h;
  z = z_h;

  nodelist_h.resize(padded_domElems*8);

  // embed hexehedral elements in nodal point lattice 
  Index_t zidx = 0 ;
  nidx = 0 ;
  for (Index_t plane=0; plane<edgeElems; ++plane) {
    for (Index_t row=0; row<edgeElems; ++row) {
      for (Index_t col=0; col<edgeElems; ++col) {
        nodelist_h[0*padded_domElems+zidx] = nidx                                       ;
        nodelist_h[1*padded_domElems+zidx] = nidx                                   + 1 ;
        nodelist_h[2*padded_domElems+zidx] = nidx                       + edgeNodes + 1 ;
        nodelist_h[3*padded_domElems+zidx] = nidx                       + edgeNodes     ;
        nodelist_h[4*padded_domElems+zidx] = nidx + edgeNodes*edgeNodes                 ;
        nodelist_h[5*padded_domElems+zidx] = nidx + edgeNodes*edgeNodes             + 1 ;
        nodelist_h[6*padded_domElems+zidx] = nidx + edgeNodes*edgeNodes + edgeNodes + 1 ;
        nodelist_h[7*padded_domElems+zidx] = nidx + edgeNodes*edgeNodes + edgeNodes     ;
        ++zidx ;
        ++nidx ;
      }
      ++nidx ;
    }
    nidx += edgeNodes ;
  }

  nodelist = nodelist_h;
}


Domain *NewDomain(char* argv[], Int_t numRanks, Index_t colLoc,
               Index_t rowLoc, Index_t planeLoc,
               Index_t nx, int tp, bool structured, Int_t nr, Int_t balance, Int_t cost)
{

  Domain *domain = new Domain ;

  domain->max_streams = 32;
  domain->streams.resize(domain->max_streams);

  for (Int_t i=0;i<domain->max_streams;i++)
    cudaStreamCreate(&(domain->streams[i]));

  cudaEventCreateWithFlags(&domain->time_constraint_computed,cudaEventDisableTiming);

  Index_t domElems;
  Index_t domNodes;
  Index_t padded_domElems;

  Vector_h<Index_t> nodelist_h;
  Vector_h<Real_t> x_h;
  Vector_h<Real_t> y_h;
  Vector_h<Real_t> z_h;

  if (structured)
  {
    domain->m_tp       = tp ;
    domain->m_numRanks = numRanks ;

    domain->m_colLoc   =   colLoc ;
    domain->m_rowLoc   =   rowLoc ;
    domain->m_planeLoc = planeLoc ;

    Index_t edgeElems = nx ;
    Index_t edgeNodes = edgeElems+1 ;

    domain->sizeX = edgeElems ;
    domain->sizeY = edgeElems ;
    domain->sizeZ = edgeElems ;  

    domain->numElem = domain->sizeX*domain->sizeY*domain->sizeZ ;
    domain->padded_numElem = PAD(domain->numElem,32);

    domain->numNode = (domain->sizeX+1)*(domain->sizeY+1)*(domain->sizeZ+1) ;
    domain->padded_numNode = PAD(domain->numNode,32);

    domElems = domain->numElem ;
    domNodes = domain->numNode ;
    padded_domElems = domain->padded_numElem ;

    AllocateElemPersistent(domain,domElems,padded_domElems);
    AllocateNodalPersistent(domain,domNodes);

    domain->SetupCommBuffers(edgeNodes);

    InitializeFields(domain);

    domain->BuildMesh(nx, edgeNodes, edgeElems, domNodes, padded_domElems, x_h, y_h, z_h, nodelist_h);

    domain->numSymmX = domain->numSymmY = domain->numSymmZ = 0;

    if (domain->m_colLoc == 0) 
      domain->numSymmX = (edgeElems+1)*(edgeElems+1) ;
    if (domain->m_rowLoc == 0) 
      domain->numSymmY = (edgeElems+1)*(edgeElems+1) ;
    if (domain->m_planeLoc == 0)
      domain->numSymmZ = (edgeElems+1)*(edgeElems+1) ;

    AllocateSymmX(domain,edgeNodes*edgeNodes);
    AllocateSymmY(domain,edgeNodes*edgeNodes);
    AllocateSymmZ(domain,edgeNodes*edgeNodes);

    /* set up symmetry nodesets */

    Vector_h<Index_t> symmX_h(domain->symmX.size());
    Vector_h<Index_t> symmY_h(domain->symmY.size());
    Vector_h<Index_t> symmZ_h(domain->symmZ.size());

    Int_t nidx = 0 ;
    for (Index_t i=0; i<edgeNodes; ++i) {
       Index_t planeInc = i*edgeNodes*edgeNodes ;
       Index_t rowInc   = i*edgeNodes ;
       for (Index_t j=0; j<edgeNodes; ++j) {
         if (domain->m_planeLoc == 0) {
           symmZ_h[nidx] = rowInc   + j ;
         } 
         if (domain->m_rowLoc == 0) {
           symmY_h[nidx] = planeInc + j ;
         }
         if (domain->m_colLoc == 0) {
           symmX_h[nidx] = planeInc + j*edgeNodes ;
         }
        ++nidx ;
       }
    }

    if (domain->m_planeLoc == 0)
      domain->symmZ = symmZ_h;
    if (domain->m_rowLoc == 0)
      domain->symmY = symmY_h;
    if (domain->m_colLoc == 0)
      domain->symmX = symmX_h;

    SetupConnectivityBC(domain, edgeElems);
  }
  else
  {
    FILE *fp;
    int ee, en;

    if ((fp = fopen(argv[2], "r")) == 0) {
       printf("could not open file %s\n", argv[2]) ;
       exit( LFileError ) ;
    }

    bool fsuccess;
    fsuccess = fscanf(fp, "%d %d", &ee, &en) ;
    domain->numElem = Index_t(ee);
    domain->padded_numElem = PAD(domain->numElem,32);

    domain->numNode = Index_t(en);
    domain->padded_numNode = PAD(domain->numNode,32);

    domElems = domain->numElem ;
    domNodes = domain->numNode ;
    padded_domElems = domain->padded_numElem ;

    AllocateElemPersistent(domain,domElems,padded_domElems);
    AllocateNodalPersistent(domain,domNodes);

    InitializeFields(domain);

    /* initialize nodal coordinates */
    x_h.resize(domNodes);
    y_h.resize(domNodes);
    z_h.resize(domNodes);

    for (Index_t i=0; i<domNodes; ++i) {
       double px, py, pz ;
       fsuccess = fscanf(fp, "%lf %lf %lf", &px, &py, &pz) ;
       x_h[i] = Real_t(px) ;
       y_h[i] = Real_t(py) ;
       z_h[i] = Real_t(pz) ;
    }
    domain->x = x_h;
    domain->y = y_h;
    domain->z = z_h;

    /* embed hexehedral elements in nodal point lattice */
    nodelist_h.resize(padded_domElems*8);
    for (Index_t zidx=0; zidx<domElems; ++zidx) {
       for (Index_t ni=0; ni<Index_t(8); ++ni) {
          int n ;
          fsuccess = fscanf(fp, "%d", &n) ;
          nodelist_h[ni*padded_domElems+zidx] = Index_t(n);
       }
    }
    domain->nodelist = nodelist_h;

    /* set up face-based element neighbors */
    Vector_h<Index_t> lxim_h(domElems);
    Vector_h<Index_t> lxip_h(domElems);
    Vector_h<Index_t> letam_h(domElems);
    Vector_h<Index_t> letap_h(domElems);
    Vector_h<Index_t> lzetam_h(domElems);
    Vector_h<Index_t> lzetap_h(domElems);

    for (Index_t i=0; i<domElems; ++i) {
       int xi_m, xi_p, eta_m, eta_p, zeta_m, zeta_p ;
       fsuccess = fscanf(fp, "%d %d %d %d %d %d",
             &xi_m, &xi_p, &eta_m, &eta_p, &zeta_m, &zeta_p) ;

       lxim_h[i]   = Index_t(xi_m) ;
       lxip_h[i]   = Index_t(xi_p) ;
       letam_h[i]  = Index_t(eta_m) ;
       letap_h[i]  = Index_t(eta_p) ;
       lzetam_h[i] = Index_t(zeta_m) ;
       lzetap_h[i] = Index_t(zeta_p) ;
    }

    domain->lxim = lxim_h;
    domain->lxip = lxip_h;
    domain->letam = letam_h;
    domain->letap = letap_h;
    domain->lzetam = lzetam_h;
    domain->lzetap = lzetap_h;

    /* set up X symmetry nodeset */

    fsuccess = fscanf(fp, "%d", &domain->numSymmX) ;
    Vector_h<Index_t> symmX_h(domain->numSymmX);
    for (Index_t i=0; i<domain->numSymmX; ++i) {
       int n ;
       fsuccess = fscanf(fp, "%d", &n) ;
       symmX_h[i] = Index_t(n) ;
    }
    domain->symmX = symmX_h;

    fsuccess = fscanf(fp, "%d", &domain->numSymmY) ;
    Vector_h<Index_t> symmY_h(domain->numSymmY);
    for (Index_t i=0; i<domain->numSymmY; ++i) {
       int n ;
       fsuccess = fscanf(fp, "%d", &n) ;
       symmY_h[i] = Index_t(n) ;
    }
    domain->symmY = symmY_h;

    fsuccess = fscanf(fp, "%d", &domain->numSymmZ) ;
    Vector_h<Index_t> symmZ_h(domain->numSymmZ);
    for (Index_t i=0; i<domain->numSymmZ; ++i) {
       int n ;
       fsuccess = fscanf(fp, "%d", &n) ;
       symmZ_h[i] = Index_t(n) ;
    }
    domain->symmZ = symmZ_h;

    /* set up free surface nodeset */
    Index_t numFreeSurf;
    fsuccess = fscanf(fp, "%d", &numFreeSurf) ;
    Vector_h<Index_t> freeSurf_h(numFreeSurf);
    for (Index_t i=0; i<numFreeSurf; ++i) {
       int n ;
       fsuccess = fscanf(fp, "%d", &n) ;
       freeSurf_h[i] = Index_t(n) ;
    }
    printf("%c\n",fsuccess);//nothing
    fclose(fp);

    /* set up boundary condition information */
    Vector_h<Index_t> elemBC_h(domElems);
    Vector_h<Index_t> surfaceNode_h(domNodes);

    for (Index_t i=0; i<domain->numElem; ++i) {
       elemBC_h[i] = 0 ;
    }

    for (Index_t i=0; i<domain->numNode; ++i) {
       surfaceNode_h[i] = 0 ;
    }

    for (Index_t i=0; i<domain->numSymmX; ++i) {
       surfaceNode_h[symmX_h[i]] = 1 ;
    }

    for (Index_t i=0; i<domain->numSymmY; ++i) {
       surfaceNode_h[symmY_h[i]] = 1 ;
    }

    for (Index_t i=0; i<domain->numSymmZ; ++i) {
       surfaceNode_h[symmZ_h[i]] = 1 ;
    }

    for (Index_t zidx=0; zidx<domain->numElem; ++zidx) {
       Int_t mask = 0 ;

       for (Index_t ni=0; ni<8; ++ni) {
          mask |= (surfaceNode_h[nodelist_h[ni*domain->padded_numElem+zidx]] << ni) ;
       }

      if ((mask & 0x0f) == 0x0f) elemBC_h[zidx] |= ZETA_M_SYMM ;
      if ((mask & 0xf0) == 0xf0) elemBC_h[zidx] |= ZETA_P_SYMM ;
      if ((mask & 0x33) == 0x33) elemBC_h[zidx] |= ETA_M_SYMM ;
      if ((mask & 0xcc) == 0xcc) elemBC_h[zidx] |= ETA_P_SYMM ;
      if ((mask & 0x99) == 0x99) elemBC_h[zidx] |= XI_M_SYMM ;
      if ((mask & 0x66) == 0x66) elemBC_h[zidx] |= XI_P_SYMM ;
    }

    for (Index_t zidx=0; zidx<domain->numElem; ++zidx) {
       if (elemBC_h[zidx] == (XI_M_SYMM | ETA_M_SYMM | ZETA_M_SYMM)) {
          domain->octantCorner = zidx ;
          break ;
       }
    }

    for (Index_t i=0; i<domain->numNode; ++i) {
       surfaceNode_h[i] = 0 ;
    }

    for (Index_t i=0; i<numFreeSurf; ++i) {
       surfaceNode_h[freeSurf_h[i]] = 1 ;
    }

    for (Index_t zidx=0; zidx<domain->numElem; ++zidx) {
       Int_t mask = 0 ;

       for (Index_t ni=0; ni<8; ++ni) {
          mask |= (surfaceNode_h[nodelist_h[ni*domain->padded_numElem+zidx]] << ni) ;
       }

      if ((mask & 0x0f) == 0x0f) elemBC_h[zidx] |= ZETA_M_SYMM ;
      if ((mask & 0xf0) == 0xf0) elemBC_h[zidx] |= ZETA_P_SYMM ;
      if ((mask & 0x33) == 0x33) elemBC_h[zidx] |= ETA_M_SYMM ;
      if ((mask & 0xcc) == 0xcc) elemBC_h[zidx] |= ETA_P_SYMM ;
      if ((mask & 0x99) == 0x99) elemBC_h[zidx] |= XI_M_SYMM ;
      if ((mask & 0x66) == 0x66) elemBC_h[zidx] |= XI_P_SYMM ;
    }

    domain->elemBC = elemBC_h;

    /* deposit energy */
    domain->e[domain->octantCorner] = Real_t(3.948746e+7) ;

  }

  /* set up node-centered indexing of elements */
  Vector_h<Index_t> nodeElemCount_h(domNodes);

  for (Index_t i=0; i<domNodes; ++i) {
     nodeElemCount_h[i] = 0 ;
  }

  for (Index_t i=0; i<domElems; ++i) {
     for (Index_t j=0; j < 8; ++j) {
        ++(nodeElemCount_h[nodelist_h[j*padded_domElems+i]]);
     }
  }

  Vector_h<Index_t> nodeElemStart_h(domNodes);

  nodeElemStart_h[0] = 0;
  for (Index_t i=1; i < domNodes; ++i) {
     nodeElemStart_h[i] =
        nodeElemStart_h[i-1] + nodeElemCount_h[i-1] ;
  }
  
  Vector_h<Index_t> nodeElemCornerList_h(nodeElemStart_h[domNodes-1] +
                 nodeElemCount_h[domNodes-1] );

  for (Index_t i=0; i < domNodes; ++i) {
     nodeElemCount_h[i] = 0;
  }

  for (Index_t j=0; j < 8; ++j) {
    for (Index_t i=0; i < domElems; ++i) {
        Index_t m = nodelist_h[padded_domElems*j+i];
        Index_t k = padded_domElems*j + i ;
        Index_t offset = nodeElemStart_h[m] +
                         nodeElemCount_h[m] ;
        nodeElemCornerList_h[offset] = k;
        ++(nodeElemCount_h[m]) ;
     }
  }

  Index_t clSize = nodeElemStart_h[domNodes-1] +
                   nodeElemCount_h[domNodes-1] ;
  for (Index_t i=0; i < clSize; ++i) {
     Index_t clv = nodeElemCornerList_h[i] ;
     if ((clv < 0) || (clv > padded_domElems*8)) {
          fprintf(stderr,
   "AllocateNodeElemIndexes(): nodeElemCornerList entry out of range!\n");
          exit(1);
     }
  }

  domain->nodeElemStart = nodeElemStart_h;
  domain->nodeElemCount = nodeElemCount_h;
  domain->nodeElemCornerList = nodeElemCornerList_h;

  /* Create a material IndexSet (entire domain same material for now) */
  Vector_h<Index_t> matElemlist_h(domElems);
  for (Index_t i=0; i<domElems; ++i) {
     matElemlist_h[i] = i ;
  }
  domain->matElemlist = matElemlist_h;

  cudaMallocHost(&domain->dtcourant_h,sizeof(Real_t),0);
  cudaMallocHost(&domain->dthydro_h,sizeof(Real_t),0);
  cudaMallocHost(&domain->bad_vol_h,sizeof(Index_t),0);
  cudaMallocHost(&domain->bad_q_h,sizeof(Index_t),0);
 
  *(domain->bad_vol_h)=-1;
  *(domain->bad_q_h)=-1;
  *(domain->dthydro_h)=1e20;
  *(domain->dtcourant_h)=1e20;

  /* initialize material parameters */
  domain->time_h      = Real_t(0.) ;
  domain->dtfixed = Real_t(-1.0e-6) ;
  domain->deltatimemultlb = Real_t(1.1) ;
  domain->deltatimemultub = Real_t(1.2) ;
  domain->stoptime  = Real_t(1.0e-2) ;
  domain->dtmax     = Real_t(1.0e-2) ;
  domain->cycle   = 0 ;

  domain->e_cut = Real_t(1.0e-7) ;
  domain->p_cut = Real_t(1.0e-7) ;
  domain->q_cut = Real_t(1.0e-7) ;
  domain->u_cut = Real_t(1.0e-7) ;
  domain->v_cut = Real_t(1.0e-10) ;

  domain->hgcoef      = Real_t(3.0) ;
  domain->ss4o3       = Real_t(4.0)/Real_t(3.0) ;

  domain->qstop              =  Real_t(1.0e+12) ;
  domain->monoq_max_slope    =  Real_t(1.0) ;
  domain->monoq_limiter_mult =  Real_t(2.0) ;
  domain->qlc_monoq          = Real_t(0.5) ;
  domain->qqc_monoq          = Real_t(2.0)/Real_t(3.0) ;
  domain->qqc                = Real_t(2.0) ;

  domain->pmin =  Real_t(0.) ;
  domain->emin = Real_t(-1.0e+15) ;

  domain->dvovmax =  Real_t(0.1) ;

  domain->eosvmax =  Real_t(1.0e+9) ;
  domain->eosvmin =  Real_t(1.0e-9) ;

  domain->refdens =  Real_t(1.0) ;

  /* initialize field data */
  Vector_h<Real_t> nodalMass_h(domNodes);
  Vector_h<Real_t> volo_h(domElems);
  Vector_h<Real_t> elemMass_h(domElems);

  for (Index_t i=0; i<domElems; ++i) {
     Real_t x_local[8], y_local[8], z_local[8] ;
     for( Index_t lnode=0 ; lnode<8 ; ++lnode )
     {
       Index_t gnode = nodelist_h[lnode*padded_domElems+i];
       x_local[lnode] = x_h[gnode];
       y_local[lnode] = y_h[gnode];
       z_local[lnode] = z_h[gnode];
     }

     // volume calculations
     Real_t volume = CalcElemVolume(x_local, y_local, z_local );
     volo_h[i] = volume ;
     elemMass_h[i] = volume ;
     for (Index_t j=0; j<8; ++j) {
        Index_t gnode = nodelist_h[j*padded_domElems+i];
        nodalMass_h[gnode] += volume / Real_t(8.0) ;
     }
  }

  domain->nodalMass = nodalMass_h;
  domain->volo = volo_h;
  domain->elemMass= elemMass_h;

   /* deposit energy */
   domain->octantCorner = 0;
  // deposit initial energy
  // An energy of 3.948746e+7 is correct for a problem with
  // 45 zones along a side - we need to scale it
  const Real_t ebase = 3.948746e+7;
  Real_t scale = (nx*domain->m_tp)/45.0;
  Real_t einit = ebase*scale*scale*scale;
  //Real_t einit = ebase;
  if (domain->m_rowLoc + domain->m_colLoc + domain->m_planeLoc == 0) {
     // Dump into the first zone (which we know is in the corner)
     // of the domain that sits at the origin
       domain->e[0] = einit;
  }

  //set initial deltatime base on analytic CFL calculation
  domain->deltatime_h = (.5*cbrt(domain->volo[0]))/sqrt(2*einit);

  domain->cost = cost;
  domain->regNumList.resize(domain->numElem) ;  // material indexset
  domain->regElemlist.resize(domain->numElem) ;  // material indexset
  domain->regCSR.resize(nr);
  domain->regReps.resize(nr);
  domain->regSorted.resize(nr);

  // Setup region index sets. For now, these are constant sized
  // throughout the run, but could be changed every cycle to 
  // simulate effects of ALE on the lagrange solver

  domain->CreateRegionIndexSets(nr, balance);

  return domain ;
}

/*******************	to support region	*********************/
void Domain::sortRegions(Vector_h<Int_t>& regReps_h, Vector_h<Index_t>& regSorted_h)
{
  Index_t temp;
  Vector_h<Index_t> regIndex;
  regIndex.resize(numReg);
  for(int i = 0; i < numReg; i++)
	regIndex[i] = i;

  for(int i = 0; i < numReg-1; i++)
	for(int j = 0; j < numReg-i-1; j++)
		if(regReps_h[j] < regReps_h[j+1])
		{
                  temp = regReps_h[j];
                  regReps_h[j] = regReps_h[j+1];
                  regReps_h[j+1] = temp;

		  temp = regElemSize[j];
		  regElemSize[j] = regElemSize[j+1];
		  regElemSize[j+1] = temp;

                  temp = regIndex[j];
                  regIndex[j] = regIndex[j+1];
                  regIndex[j+1] = temp;
		}
  for(int i = 0; i < numReg; i++)
        regSorted_h[regIndex[i]] = i;
}

// simple function for int pow x^y, y >= 0
static Int_t POW(Int_t x, Int_t y)
{
  Int_t res = 1;
  for (Int_t i = 0; i < y; i++)
    res *= x;
  return res;
}

void Domain::CreateRegionIndexSets(Int_t nr, Int_t b)
{
#if USE_MPI   
   Index_t myRank;
   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;
   srand(myRank);
#else
   srand(0);
   Index_t myRank = 0;
#endif
   numReg = nr;
   balance = b;

   regElemSize = new Int_t[numReg];
   Index_t nextIndex = 0;

   Vector_h<Int_t> regCSR_h(regCSR.size());  // records the begining and end of each region
   Vector_h<Int_t> regReps_h(regReps.size()); // records the rep number per region
   Vector_h<Index_t> regNumList_h(regNumList.size());    // Region number per domain element
   Vector_h<Index_t> regElemlist_h(regElemlist.size());  // region indexset 
   Vector_h<Index_t> regSorted_h(regSorted.size()); // keeps index of sorted regions

   //if we only have one region just fill it
   // Fill out the regNumList with material numbers, which are always
   // the region index plus one 
   if(numReg == 1) {
      while (nextIndex < numElem) {
         regNumList_h[nextIndex] = 1;
         nextIndex++;
      }
      regElemSize[0] = 0;
   }
   //If we have more than one region distribute the elements.
   else {
      Int_t regionNum;
      Int_t regionVar;
      Int_t lastReg = -1;
      Int_t binSize;
      Int_t elements;
      Index_t runto = 0;
      Int_t costDenominator = 0;
      Int_t* regBinEnd = new Int_t[numReg];
      //Determine the relative weights of all the regions.
      for (Index_t i=0 ; i<numReg ; ++i) {
         regElemSize[i] = 0;
         costDenominator += POW((i+1), balance);  //Total cost of all regions
         regBinEnd[i] = costDenominator;  //Chance of hitting a given region is (regBinEnd[i] - regBinEdn[i-1])/costDenominator
      }
      //Until all elements are assigned
      while (nextIndex < numElem) {
         //pick the region
         regionVar = rand() % costDenominator;
         Index_t i = 0;
         while(regionVar >= regBinEnd[i])
            i++;
         //rotate the regions based on MPI rank.  Rotation is Rank % NumRegions
         regionNum = ((i + myRank) % numReg) + 1;
         // make sure we don't pick the same region twice in a row
         while(regionNum == lastReg) {
            regionVar = rand() % costDenominator;
            i = 0;
            while(regionVar >= regBinEnd[i])
               i++;
            regionNum = ((i + myRank) % numReg) + 1;
         }
         //Pick the bin size of the region and determine the number of elements.
         binSize = rand() % 1000;
         if(binSize < 773) {
           elements = rand() % 15 + 1;
         }
         else if(binSize < 937) {
           elements = rand() % 16 + 16;
         }
         else if(binSize < 970) {
           elements = rand() % 32 + 32;
         }
         else if(binSize < 974) {
           elements = rand() % 64 + 64;
         }
         else if(binSize < 978) {
           elements = rand() % 128 + 128;
         }
         else if(binSize < 981) {
           elements = rand() % 256 + 256;
         }
         else
            elements = rand() % 1537 + 512;
         runto = elements + nextIndex;
         //Store the elements.  If we hit the end before we run out of elements then just stop.
         while (nextIndex < runto && nextIndex < numElem) {
            regNumList_h[nextIndex] = regionNum;
            nextIndex++;
         }
         lastReg = regionNum;
      }
   }
   // Convert regNumList to region index sets
   // First, count size of each region 
   for (Index_t i=0 ; i<numElem ; ++i) {
      int r = regNumList_h[i]-1; // region index == regnum-1
      regElemSize[r]++;
   }

   Index_t rep;
   // Second, allocate each region index set
   for (Index_t r=0; r<numReg ; ++r) {
       if(r < numReg/2)
         rep = 1;
       else if(r < (numReg - (numReg+15)/20))
         rep = 1 + cost;
       else
         rep = 10 * (1+ cost);
       regReps_h[r] = rep;
   }

   sortRegions(regReps_h, regSorted_h);

   regCSR_h[0] = 0;
   // Second, allocate each region index set
   for (Index_t i=1 ; i<numReg ; ++i) {
      regCSR_h[i] = regCSR_h[i-1] + regElemSize[i-1];
   }

   // Third, fill index sets
   for (Index_t i=0 ; i<numElem ; ++i) {
      Index_t r = regSorted_h[regNumList_h[i]-1];       // region index == regnum-1
      regElemlist_h[regCSR_h[r]] = i;
      regCSR_h[r]++;
   }

   // Copy to device
   regCSR =  regCSR_h;  // records the begining and end of each region
   regReps =  regReps_h; // records the rep number per region
   regNumList =  regNumList_h;    // Region number per domain element
   regElemlist = regElemlist_h;  // region indexset 
   regSorted = regSorted_h; // keeps index of sorted regions

} // end of create function

static inline
void TimeIncrement(Domain* domain)
{

    // To make sure dtcourant and dthydro have been updated on host
    cudaEventSynchronize(domain->time_constraint_computed);

    Real_t targetdt = domain->stoptime - domain->time_h;

    if ((domain->dtfixed <= Real_t(0.0)) && (domain->cycle != Int_t(0))) {

      Real_t ratio ;

      /* This will require a reduction in parallel */
      Real_t gnewdt = Real_t(1.0e+20) ;
      Real_t newdt;
      if ( *(domain->dtcourant_h) < gnewdt) { 
         gnewdt = *(domain->dtcourant_h) / Real_t(2.0) ;
      }
      if ( *(domain->dthydro_h) < gnewdt) { 
         gnewdt = *(domain->dthydro_h) * Real_t(2.0) / Real_t(3.0) ;
      }

#if USE_MPI      
      MPI_Allreduce(&gnewdt, &newdt, 1,
                    ((sizeof(Real_t) == 4) ? MPI_FLOAT : MPI_DOUBLE),
                    MPI_MIN, MPI_COMM_WORLD) ;
#else
      newdt = gnewdt;
#endif

      Real_t olddt = domain->deltatime_h;
      ratio = newdt / olddt ;
      if (ratio >= Real_t(1.0)) {
         if (ratio < domain->deltatimemultlb) {
            newdt = olddt ;
         }
         else if (ratio > domain->deltatimemultub) {
            newdt = olddt*domain->deltatimemultub ;
         }
      }

      if (newdt > domain->dtmax) {
         newdt = domain->dtmax ;
      }
      domain->deltatime_h = newdt ;
   }

   /* TRY TO PREVENT VERY SMALL SCALING ON THE NEXT CYCLE */
   if ((targetdt > domain->deltatime_h) &&
       (targetdt < (Real_t(4.0) * domain->deltatime_h / Real_t(3.0))) ) {
      targetdt = Real_t(2.0) * domain->deltatime_h / Real_t(3.0) ;
   }

   if (targetdt < domain->deltatime_h) {
      domain->deltatime_h = targetdt ;
   }

   domain->time_h += domain->deltatime_h ;

   ++domain->cycle ;
}

__device__
static 
 __forceinline__
void CalcElemShapeFunctionDerivatives( const Real_t* const x,
                                       const Real_t* const y,
                                       const Real_t* const z,
                                       Real_t b[][8],
                                       Real_t* const volume )
{
  const Real_t x0 = x[0] ;   const Real_t x1 = x[1] ;
  const Real_t x2 = x[2] ;   const Real_t x3 = x[3] ;
  const Real_t x4 = x[4] ;   const Real_t x5 = x[5] ;
  const Real_t x6 = x[6] ;   const Real_t x7 = x[7] ;

  const Real_t y0 = y[0] ;   const Real_t y1 = y[1] ;
  const Real_t y2 = y[2] ;   const Real_t y3 = y[3] ;
  const Real_t y4 = y[4] ;   const Real_t y5 = y[5] ;
  const Real_t y6 = y[6] ;   const Real_t y7 = y[7] ;

  const Real_t z0 = z[0] ;   const Real_t z1 = z[1] ;
  const Real_t z2 = z[2] ;   const Real_t z3 = z[3] ;
  const Real_t z4 = z[4] ;   const Real_t z5 = z[5] ;
  const Real_t z6 = z[6] ;   const Real_t z7 = z[7] ;

  Real_t fjxxi, fjxet, fjxze;
  Real_t fjyxi, fjyet, fjyze;
  Real_t fjzxi, fjzet, fjzze;
  Real_t cjxxi, cjxet, cjxze;
  Real_t cjyxi, cjyet, cjyze;
  Real_t cjzxi, cjzet, cjzze;

  fjxxi = Real_t(.125) * ( (x6-x0) + (x5-x3) - (x7-x1) - (x4-x2) );
  fjxet = Real_t(.125) * ( (x6-x0) - (x5-x3) + (x7-x1) - (x4-x2) );
  fjxze = Real_t(.125) * ( (x6-x0) + (x5-x3) + (x7-x1) + (x4-x2) );

  fjyxi = Real_t(.125) * ( (y6-y0) + (y5-y3) - (y7-y1) - (y4-y2) );
  fjyet = Real_t(.125) * ( (y6-y0) - (y5-y3) + (y7-y1) - (y4-y2) );
  fjyze = Real_t(.125) * ( (y6-y0) + (y5-y3) + (y7-y1) + (y4-y2) );

  fjzxi = Real_t(.125) * ( (z6-z0) + (z5-z3) - (z7-z1) - (z4-z2) );
  fjzet = Real_t(.125) * ( (z6-z0) - (z5-z3) + (z7-z1) - (z4-z2) );
  fjzze = Real_t(.125) * ( (z6-z0) + (z5-z3) + (z7-z1) + (z4-z2) );

  /* compute cofactors */
  cjxxi =    (fjyet * fjzze) - (fjzet * fjyze);
  cjxet =  - (fjyxi * fjzze) + (fjzxi * fjyze);
  cjxze =    (fjyxi * fjzet) - (fjzxi * fjyet);

  cjyxi =  - (fjxet * fjzze) + (fjzet * fjxze);
  cjyet =    (fjxxi * fjzze) - (fjzxi * fjxze);
  cjyze =  - (fjxxi * fjzet) + (fjzxi * fjxet);

  cjzxi =    (fjxet * fjyze) - (fjyet * fjxze);
  cjzet =  - (fjxxi * fjyze) + (fjyxi * fjxze);
  cjzze =    (fjxxi * fjyet) - (fjyxi * fjxet);

  /* calculate partials :
     this need only be done for l = 0,1,2,3   since , by symmetry ,
     (6,7,4,5) = - (0,1,2,3) .
  */
  b[0][0] =   -  cjxxi  -  cjxet  -  cjxze;
  b[0][1] =      cjxxi  -  cjxet  -  cjxze;
  b[0][2] =      cjxxi  +  cjxet  -  cjxze;
  b[0][3] =   -  cjxxi  +  cjxet  -  cjxze;
  b[0][4] = -b[0][2];
  b[0][5] = -b[0][3];
  b[0][6] = -b[0][0];
  b[0][7] = -b[0][1];

  /*

  b[0][4] = - cjxxi  -  cjxet  +  cjxze;
  b[0][5] = + cjxxi  -  cjxet  +  cjxze;
  b[0][6] = + cjxxi  +  cjxet  +  cjxze;
  b[0][7] = - cjxxi  +  cjxet  +  cjxze;

  */

  b[1][0] =   -  cjyxi  -  cjyet  -  cjyze;
  b[1][1] =      cjyxi  -  cjyet  -  cjyze;
  b[1][2] =      cjyxi  +  cjyet  -  cjyze;
  b[1][3] =   -  cjyxi  +  cjyet  -  cjyze;
  b[1][4] = -b[1][2];
  b[1][5] = -b[1][3];
  b[1][6] = -b[1][0];
  b[1][7] = -b[1][1];

  b[2][0] =   -  cjzxi  -  cjzet  -  cjzze;
  b[2][1] =      cjzxi  -  cjzet  -  cjzze;
  b[2][2] =      cjzxi  +  cjzet  -  cjzze;
  b[2][3] =   -  cjzxi  +  cjzet  -  cjzze;
  b[2][4] = -b[2][2];
  b[2][5] = -b[2][3];
  b[2][6] = -b[2][0];
  b[2][7] = -b[2][1];

  /* calculate jacobian determinant (volume) */
  *volume = Real_t(8.) * ( fjxet * cjxet + fjyet * cjyet + fjzet * cjzet);
}

static
__device__
__forceinline__
void SumElemFaceNormal(Real_t *normalX0, Real_t *normalY0, Real_t *normalZ0,
                       Real_t *normalX1, Real_t *normalY1, Real_t *normalZ1,
                       Real_t *normalX2, Real_t *normalY2, Real_t *normalZ2,
                       Real_t *normalX3, Real_t *normalY3, Real_t *normalZ3,
                       const Real_t x0, const Real_t y0, const Real_t z0,
                       const Real_t x1, const Real_t y1, const Real_t z1,
                       const Real_t x2, const Real_t y2, const Real_t z2,
                       const Real_t x3, const Real_t y3, const Real_t z3)
{
   Real_t bisectX0 = Real_t(0.5) * (x3 + x2 - x1 - x0);
   Real_t bisectY0 = Real_t(0.5) * (y3 + y2 - y1 - y0);
   Real_t bisectZ0 = Real_t(0.5) * (z3 + z2 - z1 - z0);
   Real_t bisectX1 = Real_t(0.5) * (x2 + x1 - x3 - x0);
   Real_t bisectY1 = Real_t(0.5) * (y2 + y1 - y3 - y0);
   Real_t bisectZ1 = Real_t(0.5) * (z2 + z1 - z3 - z0);
   Real_t areaX = Real_t(0.25) * (bisectY0 * bisectZ1 - bisectZ0 * bisectY1);
   Real_t areaY = Real_t(0.25) * (bisectZ0 * bisectX1 - bisectX0 * bisectZ1);
   Real_t areaZ = Real_t(0.25) * (bisectX0 * bisectY1 - bisectY0 * bisectX1);

   *normalX0 += areaX;
   *normalX1 += areaX;
   *normalX2 += areaX;
   *normalX3 += areaX;

   *normalY0 += areaY;
   *normalY1 += areaY;
   *normalY2 += areaY;
   *normalY3 += areaY;

   *normalZ0 += areaZ;
   *normalZ1 += areaZ;
   *normalZ2 += areaZ;
   *normalZ3 += areaZ;
}

static
__device__
__forceinline__
void SumElemFaceNormal_warp_per_4cell(
                       Real_t *normalX0, Real_t *normalY0, Real_t *normalZ0,
                       const Real_t x, const Real_t y, const Real_t z,
                       int node, 
                       int n0, int n1, int n2, int n3)
{
  Real_t coef0 = Real_t(0.5);
  Real_t coef1 = Real_t(0.5);
  if (node == n0 || node == n1 || node==n2 || node==n3)
  {
    if (node == n0 || node == n1)
       coef0 = -coef0;

    if (node == n0 || node == n3)
       coef1 = -coef1;
  }
  else
  {
    coef0 = Real_t(0.);
    coef1 = Real_t(0.);
  }
   
   Real_t bisectX0 = coef0*x;
   Real_t bisectY0 = coef0*y;
   Real_t bisectZ0 = coef0*z;

   Real_t bisectX1 = coef1*x;
   Real_t bisectY1 = coef1*y;
   Real_t bisectZ1 = coef1*z;

   SumOverNodesShfl(bisectX0);
   SumOverNodesShfl(bisectY0);
   SumOverNodesShfl(bisectZ0);

   SumOverNodesShfl(bisectX1);
   SumOverNodesShfl(bisectY1);
   SumOverNodesShfl(bisectZ1);

   Real_t areaX = Real_t(0.25) * (bisectY0 * bisectZ1 - bisectZ0 * bisectY1);
   Real_t areaY = Real_t(0.25) * (bisectZ0 * bisectX1 - bisectX0 * bisectZ1);
   Real_t areaZ = Real_t(0.25) * (bisectX0 * bisectY1 - bisectY0 * bisectX1);

   if (node == n0 || node == n1 || node==n2 || node==n3)
   {
     *normalX0 += areaX;
     *normalY0 += areaY;
     *normalZ0 += areaZ;
   }

}

__device__
static inline
void CalcElemNodeNormals(Real_t pfx[8],
                         Real_t pfy[8],
                         Real_t pfz[8],
                         const Real_t x[8],
                         const Real_t y[8],
                         const Real_t z[8])
{
   for (Index_t i = 0 ; i < 8 ; ++i) {
      pfx[i] = Real_t(0.0);
      pfy[i] = Real_t(0.0);
      pfz[i] = Real_t(0.0);
   }
   /* evaluate face one: nodes 0, 1, 2, 3 */
   SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0],
                  &pfx[1], &pfy[1], &pfz[1],
                  &pfx[2], &pfy[2], &pfz[2],
                  &pfx[3], &pfy[3], &pfz[3],
                  x[0], y[0], z[0], x[1], y[1], z[1],
                  x[2], y[2], z[2], x[3], y[3], z[3]);
   /* evaluate face two: nodes 0, 4, 5, 1 */
   SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0],
                  &pfx[4], &pfy[4], &pfz[4],
                  &pfx[5], &pfy[5], &pfz[5],
                  &pfx[1], &pfy[1], &pfz[1],
                  x[0], y[0], z[0], x[4], y[4], z[4],
                  x[5], y[5], z[5], x[1], y[1], z[1]);
   /* evaluate face three: nodes 1, 5, 6, 2 */
   SumElemFaceNormal(&pfx[1], &pfy[1], &pfz[1],
                  &pfx[5], &pfy[5], &pfz[5],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[2], &pfy[2], &pfz[2],
                  x[1], y[1], z[1], x[5], y[5], z[5],
                  x[6], y[6], z[6], x[2], y[2], z[2]);
   /* evaluate face four: nodes 2, 6, 7, 3 */
   SumElemFaceNormal(&pfx[2], &pfy[2], &pfz[2],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[3], &pfy[3], &pfz[3],
                  x[2], y[2], z[2], x[6], y[6], z[6],
                  x[7], y[7], z[7], x[3], y[3], z[3]);
   /* evaluate face five: nodes 3, 7, 4, 0 */
   SumElemFaceNormal(&pfx[3], &pfy[3], &pfz[3],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[4], &pfy[4], &pfz[4],
                  &pfx[0], &pfy[0], &pfz[0],
                  x[3], y[3], z[3], x[7], y[7], z[7],
                  x[4], y[4], z[4], x[0], y[0], z[0]);
   /* evaluate face six: nodes 4, 7, 6, 5 */
   SumElemFaceNormal(&pfx[4], &pfy[4], &pfz[4],
                  &pfx[7], &pfy[7], &pfz[7],
                  &pfx[6], &pfy[6], &pfz[6],
                  &pfx[5], &pfy[5], &pfz[5],
                  x[4], y[4], z[4], x[7], y[7], z[7],
                  x[6], y[6], z[6], x[5], y[5], z[5]);
}



__global__
void AddNodeForcesFromElems_kernel( Index_t numNode,
                                    Index_t padded_numNode,
                                    const Int_t* nodeElemCount, 
                                    const Int_t* nodeElemStart, 
                                    const Index_t* nodeElemCornerList,
                                    const Real_t* fx_elem, 
                                    const Real_t* fy_elem, 
                                    const Real_t* fz_elem,
                                    Real_t* fx_node, 
                                    Real_t* fy_node, 
                                    Real_t* fz_node,
                                    const Int_t num_threads)
{
    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    if (tid < num_threads)
    {
      Index_t g_i = tid;
      Int_t count=nodeElemCount[g_i];
      Int_t start=nodeElemStart[g_i];
      Real_t fx,fy,fz;
      fx=fy=fz=Real_t(0.0);

      for (int j=0;j<count;j++) 
      {
          Index_t pos=nodeElemCornerList[start+j]; // Uncoalesced access here
          fx += fx_elem[pos]; 
          fy += fy_elem[pos]; 
          fz += fz_elem[pos];
      }


      fx_node[g_i]=fx; 
      fy_node[g_i]=fy; 
      fz_node[g_i]=fz;
    }
}

static
__device__
__forceinline__
void VoluDer(const Real_t x0, const Real_t x1, const Real_t x2,
             const Real_t x3, const Real_t x4, const Real_t x5,
             const Real_t y0, const Real_t y1, const Real_t y2,
             const Real_t y3, const Real_t y4, const Real_t y5,
             const Real_t z0, const Real_t z1, const Real_t z2,
             const Real_t z3, const Real_t z4, const Real_t z5,
             Real_t* dvdx, Real_t* dvdy, Real_t* dvdz)
{
   const Real_t twelfth = Real_t(1.0) / Real_t(12.0) ;

   *dvdx =
      (y1 + y2) * (z0 + z1) - (y0 + y1) * (z1 + z2) +
      (y0 + y4) * (z3 + z4) - (y3 + y4) * (z0 + z4) -
      (y2 + y5) * (z3 + z5) + (y3 + y5) * (z2 + z5);

   *dvdy =
      - (x1 + x2) * (z0 + z1) + (x0 + x1) * (z1 + z2) -
      (x0 + x4) * (z3 + z4) + (x3 + x4) * (z0 + z4) +
      (x2 + x5) * (z3 + z5) - (x3 + x5) * (z2 + z5);

   *dvdz =
      - (y1 + y2) * (x0 + x1) + (y0 + y1) * (x1 + x2) -
      (y0 + y4) * (x3 + x4) + (y3 + y4) * (x0 + x4) +
      (y2 + y5) * (x3 + x5) - (y3 + y5) * (x2 + x5);

   *dvdx *= twelfth;
   *dvdy *= twelfth;
   *dvdz *= twelfth;
}

static
__device__
__forceinline__
void CalcElemVolumeDerivative(Real_t dvdx[8],
                              Real_t dvdy[8],
                              Real_t dvdz[8],
                              const Real_t x[8],
                              const Real_t y[8],
                              const Real_t z[8])
{
   VoluDer(x[1], x[2], x[3], x[4], x[5], x[7],
           y[1], y[2], y[3], y[4], y[5], y[7],
           z[1], z[2], z[3], z[4], z[5], z[7],
           &dvdx[0], &dvdy[0], &dvdz[0]);
   VoluDer(x[0], x[1], x[2], x[7], x[4], x[6],
           y[0], y[1], y[2], y[7], y[4], y[6],
           z[0], z[1], z[2], z[7], z[4], z[6],
           &dvdx[3], &dvdy[3], &dvdz[3]);
   VoluDer(x[3], x[0], x[1], x[6], x[7], x[5],
           y[3], y[0], y[1], y[6], y[7], y[5],
           z[3], z[0], z[1], z[6], z[7], z[5],
           &dvdx[2], &dvdy[2], &dvdz[2]);
   VoluDer(x[2], x[3], x[0], x[5], x[6], x[4],
           y[2], y[3], y[0], y[5], y[6], y[4],
           z[2], z[3], z[0], z[5], z[6], z[4],
           &dvdx[1], &dvdy[1], &dvdz[1]);
   VoluDer(x[7], x[6], x[5], x[0], x[3], x[1],
           y[7], y[6], y[5], y[0], y[3], y[1],
           z[7], z[6], z[5], z[0], z[3], z[1],
           &dvdx[4], &dvdy[4], &dvdz[4]);
   VoluDer(x[4], x[7], x[6], x[1], x[0], x[2],
           y[4], y[7], y[6], y[1], y[0], y[2],
           z[4], z[7], z[6], z[1], z[0], z[2],
           &dvdx[5], &dvdy[5], &dvdz[5]);
   VoluDer(x[5], x[4], x[7], x[2], x[1], x[3],
           y[5], y[4], y[7], y[2], y[1], y[3],
           z[5], z[4], z[7], z[2], z[1], z[3],
           &dvdx[6], &dvdy[6], &dvdz[6]);
   VoluDer(x[6], x[5], x[4], x[3], x[2], x[0],
           y[6], y[5], y[4], y[3], y[2], y[0],
           z[6], z[5], z[4], z[3], z[2], z[0],
           &dvdx[7], &dvdy[7], &dvdz[7]);
}

static
__device__ 
__forceinline__
void CalcElemFBHourglassForce(Real_t *xd, Real_t *yd, Real_t *zd,  Real_t *hourgam0,
                              Real_t *hourgam1, Real_t *hourgam2, Real_t *hourgam3,
                              Real_t *hourgam4, Real_t *hourgam5, Real_t *hourgam6,
                              Real_t *hourgam7, Real_t coefficient,
                              Real_t *hgfx, Real_t *hgfy, Real_t *hgfz )
{
   Index_t i00=0;
   Index_t i01=1;
   Index_t i02=2;
   Index_t i03=3;

   Real_t h00 =
      hourgam0[i00] * xd[0] + hourgam1[i00] * xd[1] +
      hourgam2[i00] * xd[2] + hourgam3[i00] * xd[3] +
      hourgam4[i00] * xd[4] + hourgam5[i00] * xd[5] +
      hourgam6[i00] * xd[6] + hourgam7[i00] * xd[7];

   Real_t h01 =
      hourgam0[i01] * xd[0] + hourgam1[i01] * xd[1] +
      hourgam2[i01] * xd[2] + hourgam3[i01] * xd[3] +
      hourgam4[i01] * xd[4] + hourgam5[i01] * xd[5] +
      hourgam6[i01] * xd[6] + hourgam7[i01] * xd[7];

   Real_t h02 =
      hourgam0[i02] * xd[0] + hourgam1[i02] * xd[1]+
      hourgam2[i02] * xd[2] + hourgam3[i02] * xd[3]+
      hourgam4[i02] * xd[4] + hourgam5[i02] * xd[5]+
      hourgam6[i02] * xd[6] + hourgam7[i02] * xd[7];

   Real_t h03 =
      hourgam0[i03] * xd[0] + hourgam1[i03] * xd[1] +
      hourgam2[i03] * xd[2] + hourgam3[i03] * xd[3] +
      hourgam4[i03] * xd[4] + hourgam5[i03] * xd[5] +
      hourgam6[i03] * xd[6] + hourgam7[i03] * xd[7];

   hgfx[0] += coefficient *
      (hourgam0[i00] * h00 + hourgam0[i01] * h01 +
       hourgam0[i02] * h02 + hourgam0[i03] * h03);

   hgfx[1] += coefficient *
      (hourgam1[i00] * h00 + hourgam1[i01] * h01 +
       hourgam1[i02] * h02 + hourgam1[i03] * h03);

   hgfx[2] += coefficient *
      (hourgam2[i00] * h00 + hourgam2[i01] * h01 +
       hourgam2[i02] * h02 + hourgam2[i03] * h03);

   hgfx[3] += coefficient *
      (hourgam3[i00] * h00 + hourgam3[i01] * h01 +
       hourgam3[i02] * h02 + hourgam3[i03] * h03);

   hgfx[4] += coefficient *
      (hourgam4[i00] * h00 + hourgam4[i01] * h01 +
       hourgam4[i02] * h02 + hourgam4[i03] * h03);

   hgfx[5] += coefficient *
      (hourgam5[i00] * h00 + hourgam5[i01] * h01 +
       hourgam5[i02] * h02 + hourgam5[i03] * h03);

   hgfx[6] += coefficient *
      (hourgam6[i00] * h00 + hourgam6[i01] * h01 +
       hourgam6[i02] * h02 + hourgam6[i03] * h03);

   hgfx[7] += coefficient *
      (hourgam7[i00] * h00 + hourgam7[i01] * h01 +
       hourgam7[i02] * h02 + hourgam7[i03] * h03);

   h00 =
      hourgam0[i00] * yd[0] + hourgam1[i00] * yd[1] +
      hourgam2[i00] * yd[2] + hourgam3[i00] * yd[3] +
      hourgam4[i00] * yd[4] + hourgam5[i00] * yd[5] +
      hourgam6[i00] * yd[6] + hourgam7[i00] * yd[7];

   h01 =
      hourgam0[i01] * yd[0] + hourgam1[i01] * yd[1] +
      hourgam2[i01] * yd[2] + hourgam3[i01] * yd[3] +
      hourgam4[i01] * yd[4] + hourgam5[i01] * yd[5] +
      hourgam6[i01] * yd[6] + hourgam7[i01] * yd[7];

   h02 =
      hourgam0[i02] * yd[0] + hourgam1[i02] * yd[1]+
      hourgam2[i02] * yd[2] + hourgam3[i02] * yd[3]+
      hourgam4[i02] * yd[4] + hourgam5[i02] * yd[5]+
      hourgam6[i02] * yd[6] + hourgam7[i02] * yd[7];

   h03 =
      hourgam0[i03] * yd[0] + hourgam1[i03] * yd[1] +
      hourgam2[i03] * yd[2] + hourgam3[i03] * yd[3] +
      hourgam4[i03] * yd[4] + hourgam5[i03] * yd[5] +
      hourgam6[i03] * yd[6] + hourgam7[i03] * yd[7];


   hgfy[0] += coefficient *
      (hourgam0[i00] * h00 + hourgam0[i01] * h01 +
       hourgam0[i02] * h02 + hourgam0[i03] * h03);

   hgfy[1] += coefficient *
      (hourgam1[i00] * h00 + hourgam1[i01] * h01 +
       hourgam1[i02] * h02 + hourgam1[i03] * h03);

   hgfy[2] += coefficient *
      (hourgam2[i00] * h00 + hourgam2[i01] * h01 +
       hourgam2[i02] * h02 + hourgam2[i03] * h03);

   hgfy[3] += coefficient *
      (hourgam3[i00] * h00 + hourgam3[i01] * h01 +
       hourgam3[i02] * h02 + hourgam3[i03] * h03);

   hgfy[4] += coefficient *
      (hourgam4[i00] * h00 + hourgam4[i01] * h01 +
       hourgam4[i02] * h02 + hourgam4[i03] * h03);

   hgfy[5] += coefficient *
      (hourgam5[i00] * h00 + hourgam5[i01] * h01 +
       hourgam5[i02] * h02 + hourgam5[i03] * h03);

   hgfy[6] += coefficient *
      (hourgam6[i00] * h00 + hourgam6[i01] * h01 +
       hourgam6[i02] * h02 + hourgam6[i03] * h03);

   hgfy[7] += coefficient *
      (hourgam7[i00] * h00 + hourgam7[i01] * h01 +
       hourgam7[i02] * h02 + hourgam7[i03] * h03);

   h00 =
      hourgam0[i00] * zd[0] + hourgam1[i00] * zd[1] +
      hourgam2[i00] * zd[2] + hourgam3[i00] * zd[3] +
      hourgam4[i00] * zd[4] + hourgam5[i00] * zd[5] +
      hourgam6[i00] * zd[6] + hourgam7[i00] * zd[7];

   h01 =
      hourgam0[i01] * zd[0] + hourgam1[i01] * zd[1] +
      hourgam2[i01] * zd[2] + hourgam3[i01] * zd[3] +
      hourgam4[i01] * zd[4] + hourgam5[i01] * zd[5] +
      hourgam6[i01] * zd[6] + hourgam7[i01] * zd[7];

   h02 =
      hourgam0[i02] * zd[0] + hourgam1[i02] * zd[1]+
      hourgam2[i02] * zd[2] + hourgam3[i02] * zd[3]+
      hourgam4[i02] * zd[4] + hourgam5[i02] * zd[5]+
      hourgam6[i02] * zd[6] + hourgam7[i02] * zd[7];

   h03 =
      hourgam0[i03] * zd[0] + hourgam1[i03] * zd[1] +
      hourgam2[i03] * zd[2] + hourgam3[i03] * zd[3] +
      hourgam4[i03] * zd[4] + hourgam5[i03] * zd[5] +
      hourgam6[i03] * zd[6] + hourgam7[i03] * zd[7];


   hgfz[0] += coefficient *
      (hourgam0[i00] * h00 + hourgam0[i01] * h01 +
       hourgam0[i02] * h02 + hourgam0[i03] * h03);

   hgfz[1] += coefficient *
      (hourgam1[i00] * h00 + hourgam1[i01] * h01 +
       hourgam1[i02] * h02 + hourgam1[i03] * h03);

   hgfz[2] += coefficient *
      (hourgam2[i00] * h00 + hourgam2[i01] * h01 +
       hourgam2[i02] * h02 + hourgam2[i03] * h03);

   hgfz[3] += coefficient *
      (hourgam3[i00] * h00 + hourgam3[i01] * h01 +
       hourgam3[i02] * h02 + hourgam3[i03] * h03);

   hgfz[4] += coefficient *
      (hourgam4[i00] * h00 + hourgam4[i01] * h01 +
       hourgam4[i02] * h02 + hourgam4[i03] * h03);

   hgfz[5] += coefficient *
      (hourgam5[i00] * h00 + hourgam5[i01] * h01 +
       hourgam5[i02] * h02 + hourgam5[i03] * h03);

   hgfz[6] += coefficient *
      (hourgam6[i00] * h00 + hourgam6[i01] * h01 +
       hourgam6[i02] * h02 + hourgam6[i03] * h03);

   hgfz[7] += coefficient *
      (hourgam7[i00] * h00 + hourgam7[i01] * h01 +
       hourgam7[i02] * h02 + hourgam7[i03] * h03);
}

__device__
__forceinline__
void CalcHourglassModes(const Real_t xn[8], const Real_t yn[8], const Real_t zn[8],
                        const Real_t dvdxn[8], const Real_t dvdyn[8], const Real_t dvdzn[8],
                        Real_t hourgam[8][4], Real_t volinv)
{
    Real_t hourmodx, hourmody, hourmodz;

    hourmodx = xn[0] + xn[1] - xn[2] - xn[3] - xn[4] - xn[5] + xn[6] + xn[7];
    hourmody = yn[0] + yn[1] - yn[2] - yn[3] - yn[4] - yn[5] + yn[6] + yn[7];
    hourmodz = zn[0] + zn[1] - zn[2] - zn[3] - zn[4] - zn[5] + zn[6] + zn[7]; // 21
    hourgam[0][0] =  1.0 - volinv*(dvdxn[0]*hourmodx + dvdyn[0]*hourmody + dvdzn[0]*hourmodz);
    hourgam[1][0] =  1.0 - volinv*(dvdxn[1]*hourmodx + dvdyn[1]*hourmody + dvdzn[1]*hourmodz);
    hourgam[2][0] = -1.0 - volinv*(dvdxn[2]*hourmodx + dvdyn[2]*hourmody + dvdzn[2]*hourmodz);
    hourgam[3][0] = -1.0 - volinv*(dvdxn[3]*hourmodx + dvdyn[3]*hourmody + dvdzn[3]*hourmodz);
    hourgam[4][0] = -1.0 - volinv*(dvdxn[4]*hourmodx + dvdyn[4]*hourmody + dvdzn[4]*hourmodz);
    hourgam[5][0] = -1.0 - volinv*(dvdxn[5]*hourmodx + dvdyn[5]*hourmody + dvdzn[5]*hourmodz);
    hourgam[6][0] =  1.0 - volinv*(dvdxn[6]*hourmodx + dvdyn[6]*hourmody + dvdzn[6]*hourmodz);
    hourgam[7][0] =  1.0 - volinv*(dvdxn[7]*hourmodx + dvdyn[7]*hourmody + dvdzn[7]*hourmodz); // 60

    hourmodx = xn[0] - xn[1] - xn[2] + xn[3] - xn[4] + xn[5] + xn[6] - xn[7];
    hourmody = yn[0] - yn[1] - yn[2] + yn[3] - yn[4] + yn[5] + yn[6] - yn[7];
    hourmodz = zn[0] - zn[1] - zn[2] + zn[3] - zn[4] + zn[5] + zn[6] - zn[7];
    hourgam[0][1] =  1.0 - volinv*(dvdxn[0]*hourmodx + dvdyn[0]*hourmody + dvdzn[0]*hourmodz);
    hourgam[1][1] = -1.0 - volinv*(dvdxn[1]*hourmodx + dvdyn[1]*hourmody + dvdzn[1]*hourmodz);
    hourgam[2][1] = -1.0 - volinv*(dvdxn[2]*hourmodx + dvdyn[2]*hourmody + dvdzn[2]*hourmodz);
    hourgam[3][1] =  1.0 - volinv*(dvdxn[3]*hourmodx + dvdyn[3]*hourmody + dvdzn[3]*hourmodz);
    hourgam[4][1] = -1.0 - volinv*(dvdxn[4]*hourmodx + dvdyn[4]*hourmody + dvdzn[4]*hourmodz);
    hourgam[5][1] =  1.0 - volinv*(dvdxn[5]*hourmodx + dvdyn[5]*hourmody + dvdzn[5]*hourmodz);
    hourgam[6][1] =  1.0 - volinv*(dvdxn[6]*hourmodx + dvdyn[6]*hourmody + dvdzn[6]*hourmodz);
    hourgam[7][1] = -1.0 - volinv*(dvdxn[7]*hourmodx + dvdyn[7]*hourmody + dvdzn[7]*hourmodz);

    hourmodx = xn[0] - xn[1] + xn[2] - xn[3] + xn[4] - xn[5] + xn[6] - xn[7];
    hourmody = yn[0] - yn[1] + yn[2] - yn[3] + yn[4] - yn[5] + yn[6] - yn[7];
    hourmodz = zn[0] - zn[1] + zn[2] - zn[3] + zn[4] - zn[5] + zn[6] - zn[7];
    hourgam[0][2] =  1.0 - volinv*(dvdxn[0]*hourmodx + dvdyn[0]*hourmody + dvdzn[0]*hourmodz);
    hourgam[1][2] = -1.0 - volinv*(dvdxn[1]*hourmodx + dvdyn[1]*hourmody + dvdzn[1]*hourmodz);
    hourgam[2][2] =  1.0 - volinv*(dvdxn[2]*hourmodx + dvdyn[2]*hourmody + dvdzn[2]*hourmodz);
    hourgam[3][2] = -1.0 - volinv*(dvdxn[3]*hourmodx + dvdyn[3]*hourmody + dvdzn[3]*hourmodz);
    hourgam[4][2] =  1.0 - volinv*(dvdxn[4]*hourmodx + dvdyn[4]*hourmody + dvdzn[4]*hourmodz);
    hourgam[5][2] = -1.0 - volinv*(dvdxn[5]*hourmodx + dvdyn[5]*hourmody + dvdzn[5]*hourmodz);
    hourgam[6][2] =  1.0 - volinv*(dvdxn[6]*hourmodx + dvdyn[6]*hourmody + dvdzn[6]*hourmodz);
    hourgam[7][2] = -1.0 - volinv*(dvdxn[7]*hourmodx + dvdyn[7]*hourmody + dvdzn[7]*hourmodz);

    hourmodx = -xn[0] + xn[1] - xn[2] + xn[3] + xn[4] - xn[5] + xn[6] - xn[7];
    hourmody = -yn[0] + yn[1] - yn[2] + yn[3] + yn[4] - yn[5] + yn[6] - yn[7];
    hourmodz = -zn[0] + zn[1] - zn[2] + zn[3] + zn[4] - zn[5] + zn[6] - zn[7];
    hourgam[0][3] = -1.0 - volinv*(dvdxn[0]*hourmodx + dvdyn[0]*hourmody + dvdzn[0]*hourmodz);
    hourgam[1][3] =  1.0 - volinv*(dvdxn[1]*hourmodx + dvdyn[1]*hourmody + dvdzn[1]*hourmodz);
    hourgam[2][3] = -1.0 - volinv*(dvdxn[2]*hourmodx + dvdyn[2]*hourmody + dvdzn[2]*hourmodz);
    hourgam[3][3] =  1.0 - volinv*(dvdxn[3]*hourmodx + dvdyn[3]*hourmody + dvdzn[3]*hourmodz);
    hourgam[4][3] =  1.0 - volinv*(dvdxn[4]*hourmodx + dvdyn[4]*hourmody + dvdzn[4]*hourmodz);
    hourgam[5][3] = -1.0 - volinv*(dvdxn[5]*hourmodx + dvdyn[5]*hourmody + dvdzn[5]*hourmodz);
    hourgam[6][3] =  1.0 - volinv*(dvdxn[6]*hourmodx + dvdyn[6]*hourmody + dvdzn[6]*hourmodz);
    hourgam[7][3] = -1.0 - volinv*(dvdxn[7]*hourmodx + dvdyn[7]*hourmody + dvdzn[7]*hourmodz);

}

template< bool hourg_gt_zero > 
__global__
#ifdef DOUBLE_PRECISION
__launch_bounds__(64,4) 
#else
__launch_bounds__(64,8) 
#endif
void CalcVolumeForceForElems_kernel(

    const Real_t* __restrict__ volo, 
    const Real_t* __restrict__ v,
    const Real_t* __restrict__ p, 
    const Real_t* __restrict__ q,
    Real_t hourg,
    Index_t numElem, 
    Index_t padded_numElem, 
    const Index_t* __restrict__ nodelist,
    const Real_t* __restrict__ ss, 
    const Real_t* __restrict__ elemMass,
    const Real_t* __restrict__ x,   const Real_t* __restrict__ y,  const Real_t* __restrict__  z,
    const Real_t* __restrict__ xd,  const Real_t* __restrict__ yd,  const Real_t* __restrict__  zd,
    //TextureObj<Real_t> x,  TextureObj<Real_t> y,  TextureObj<Real_t> z,
    //TextureObj<Real_t> xd,  TextureObj<Real_t> yd,  TextureObj<Real_t> zd,
    //TextureObj<Real_t>* x,  TextureObj<Real_t>* y,  TextureObj<Real_t>* z,
    //TextureObj<Real_t>* xd,  TextureObj<Real_t>* yd,  TextureObj<Real_t>* zd,
#ifdef DOUBLE_PRECISION // For floats, use atomicAdd
    Real_t* __restrict__ fx_elem, 
    Real_t* __restrict__ fy_elem, 
    Real_t* __restrict__ fz_elem,
#else
    Real_t* __restrict__ fx_node, 
    Real_t* __restrict__ fy_node, 
    Real_t* __restrict__ fz_node,
#endif
    Index_t* __restrict__ bad_vol,
    const Index_t num_threads)

{

  /*************************************************
  *     FUNCTION: Calculates the volume forces
  *************************************************/

  Real_t xn[8],yn[8],zn[8];;
  Real_t xdn[8],ydn[8],zdn[8];;
  Real_t dvdxn[8],dvdyn[8],dvdzn[8];;
  Real_t hgfx[8],hgfy[8],hgfz[8];;
  Real_t hourgam[8][4];
  Real_t coefficient;

  int elem=blockDim.x*blockIdx.x+threadIdx.x;
  if (elem < num_threads) 
  {
    Real_t volume = v[elem];
    Real_t det = volo[elem] * volume;

    // Check for bad volume
    if (volume < 0.) {
      *bad_vol = elem; 
    }

    Real_t ss1 = ss[elem];
    Real_t mass1 = elemMass[elem];
    Real_t sigxx = -p[elem] - q[elem];

    Index_t n[8];
    #pragma unroll 
    for (int i=0;i<8;i++) {
      n[i] = nodelist[elem+i*padded_numElem];
    }

    Real_t volinv = Real_t(1.0) / det;
    //#pragma unroll 
    //for (int i=0;i<8;i++) {
    //  xn[i] =x[n[i]];
    //  yn[i] =y[n[i]];
    //  zn[i] =z[n[i]];
    //}

    #pragma unroll 
    for (int i=0;i<8;i++)
      xn[i] =x[n[i]];

    #pragma unroll 
    for (int i=0;i<8;i++)
      yn[i] =y[n[i]];

    #pragma unroll 
    for (int i=0;i<8;i++)
      zn[i] =z[n[i]];


    Real_t volume13 = CBRT(det);
    coefficient = - hourg * Real_t(0.01) * ss1 * mass1 / volume13;

    /*************************************************/
    /*    compute the volume derivatives             */
    /*************************************************/
    CalcElemVolumeDerivative(dvdxn, dvdyn, dvdzn, xn, yn, zn); 

    /*************************************************/
    /*    compute the hourglass modes                */
    /*************************************************/
    CalcHourglassModes(xn,yn,zn,dvdxn,dvdyn,dvdzn,hourgam,volinv);

    /*************************************************/
    /*    CalcStressForElems                         */
    /*************************************************/
    Real_t B[3][8];

    CalcElemShapeFunctionDerivatives(xn, yn, zn, B, &det); 

    CalcElemNodeNormals( B[0] , B[1], B[2], xn, yn, zn); 

    // Check for bad volume
    if (det < 0.) {
      *bad_vol = elem; 
    }

    #pragma unroll
    for (int i=0;i<8;i++)
    {
      hgfx[i] = -( sigxx*B[0][i] );
      hgfy[i] = -( sigxx*B[1][i] );
      hgfz[i] = -( sigxx*B[2][i] );
    }

    if (hourg_gt_zero)
    {
      /*************************************************/
      /*    CalcFBHourglassForceForElems               */
      /*************************************************/

//      #pragma unroll 
//      for (int i=0;i<8;i++) {
//        xdn[i] =xd[n[i]];
//        ydn[i] =yd[n[i]];
//        zdn[i] =zd[n[i]];
//      }

      #pragma unroll 
      for (int i=0;i<8;i++)
        xdn[i] =xd[n[i]];

      #pragma unroll 
      for (int i=0;i<8;i++)
        ydn[i] =yd[n[i]];

      #pragma unroll 
      for (int i=0;i<8;i++)
        zdn[i] =zd[n[i]];




      CalcElemFBHourglassForce
      ( &xdn[0],&ydn[0],&zdn[0],
	      hourgam[0],hourgam[1],hourgam[2],hourgam[3],
        hourgam[4],hourgam[5],hourgam[6],hourgam[7],
	    	coefficient,
	    	&hgfx[0],&hgfy[0],&hgfz[0]
      );

    }

#ifdef DOUBLE_PRECISION
    #pragma unroll
    for (int node=0;node<8;node++)
    {
      Index_t store_loc = elem+padded_numElem*node;
      fx_elem[store_loc]=hgfx[node]; 
      fy_elem[store_loc]=hgfy[node]; 
      fz_elem[store_loc]=hgfz[node];
    }
#else
    #pragma unroll
    for (int i=0;i<8;i++)
    {
      Index_t ni= n[i];
      atomicAdd(&fx_node[ni],hgfx[i]); 
      atomicAdd(&fy_node[ni],hgfy[i]); 
      atomicAdd(&fz_node[ni],hgfz[i]);
    } 
#endif

  } // If elem < numElem
}


template< bool hourg_gt_zero, int cta_size> 
__global__
void CalcVolumeForceForElems_kernel_warp_per_4cell(

    const Real_t* __restrict__ volo, 
    const Real_t* __restrict__ v,
    const Real_t* __restrict__ p, 
    const Real_t* __restrict__ q,
    Real_t hourg,
    Index_t numElem, 
    Index_t padded_numElem, 
    const Index_t* __restrict__ nodelist,
    const Real_t* __restrict__ ss, 
    const Real_t* __restrict__ elemMass,
    //const Real_t __restrict__ *x,  const Real_t __restrict__ *y,  const Real_t __restrict__ *z,
    //const Real_t __restrict__ *xd,  const Real_t __restrict__ *yd,  const Real_t __restrict__ *zd,
    const Real_t  *x,  const Real_t *y,  const Real_t *z,
    const Real_t  *xd,  const Real_t *yd,  const Real_t *zd,
#ifdef DOUBLE_PRECISION // For floats, use atomicAdd
    Real_t* __restrict__ fx_elem, 
    Real_t* __restrict__ fy_elem, 
    Real_t* __restrict__ fz_elem,
#else
    Real_t* __restrict__ fx_node, 
    Real_t* __restrict__ fy_node, 
    Real_t* __restrict__ fz_node,
#endif
    Index_t* __restrict__ bad_vol,
    const Index_t num_threads)

{

  /*************************************************
  *     FUNCTION: Calculates the volume forces
  *************************************************/

  Real_t xn,yn,zn;;
  Real_t xdn,ydn,zdn;;
  Real_t dvdxn,dvdyn,dvdzn;;
  Real_t hgfx,hgfy,hgfz;;
  Real_t hourgam[4];
  Real_t coefficient;

  int tid=blockDim.x*blockIdx.x+threadIdx.x;
  int elem = tid >> 3; // elem = tid/8
  int node = tid & 7; // node = tid%8

  // elem within cta
//  int cta_elem = threadIdx.x/8;

  if (elem < num_threads) 
  {
    Real_t volume = v[elem];
    Real_t det = volo[elem] * volume;

    // Check for bad volume
    if (volume < 0.) {
      *bad_vol = elem; 
    }

    Real_t ss1 = ss[elem];
    Real_t mass1 = elemMass[elem];
    Real_t sigxx = -p[elem] - q[elem];

    Index_t node_id;
    node_id = nodelist[elem+node*padded_numElem];

    Real_t volinv = Real_t(1.0) / det;

    xn =x[node_id];
    yn =y[node_id];
    zn =z[node_id];

    Real_t volume13 = CBRT(det);
    coefficient = - hourg * Real_t(0.01) * ss1 * mass1 / volume13;

    /*************************************************/
    /*    compute the volume derivatives             */
    /*************************************************/
    unsigned int ind0,ind1,ind2,ind3,ind4,ind5;

    // Use octal number to represent the indices for each node
    //ind0 = 012307456;
    //ind1 = 023016745;
    //ind2 = 030125674;
    //ind3 = 045670123;
    //ind4 = 056743012;
    //ind5 = 074561230;

    //int mask = 7u << (3*node;

    switch(node) {
    case 0:
      {ind0=1; ind1=2; ind2=3; ind3=4; ind4=5; ind5=7;
      break;}
    case 1:
      {ind0=2; ind1=3; ind2=0; ind3=5; ind4=6; ind5=4;
      break;}
    case 2:
      {ind0=3; ind1=0; ind2=1; ind3=6; ind4=7; ind5=5;
      break;}
    case 3:
      {ind0=0; ind1=1; ind2=2; ind3=7; ind4=4; ind5=6;
      break;}
    case 4:
      {ind0=7; ind1=6; ind2=5; ind3=0; ind4=3; ind5=1;
      break;}
    case 5:
      {ind0=4; ind1=7; ind2=6; ind3=1; ind4=0; ind5=2;
      break;}
    case 6:
      {ind0=5; ind1=4; ind2=7; ind3=2; ind4=1; ind5=3;
      break;}
    case 7:
      {ind0=6; ind1=5; ind2=4; ind3=3; ind4=2; ind5=0;
      break;}
    }

    VOLUDER(utils::shfl(yn,ind0,8),utils::shfl(yn,ind1,8),utils::shfl(yn,ind2,8),
            utils::shfl(yn,ind3,8),utils::shfl(yn,ind4,8),utils::shfl(yn,ind5,8),
            utils::shfl(zn,ind0,8),utils::shfl(zn,ind1,8),utils::shfl(zn,ind2,8),
            utils::shfl(zn,ind3,8),utils::shfl(zn,ind4,8),utils::shfl(zn,ind5,8),
	          dvdxn);

    VOLUDER(utils::shfl(zn,ind0,8),utils::shfl(zn,ind1,8),utils::shfl(zn,ind2,8),
            utils::shfl(zn,ind3,8),utils::shfl(zn,ind4,8),utils::shfl(zn,ind5,8),
            utils::shfl(xn,ind0,8),utils::shfl(xn,ind1,8),utils::shfl(xn,ind2,8),
            utils::shfl(xn,ind3,8),utils::shfl(xn,ind4,8),utils::shfl(xn,ind5,8),
	          dvdyn);

    VOLUDER(utils::shfl(xn,ind0,8),utils::shfl(xn,ind1,8),utils::shfl(xn,ind2,8),
            utils::shfl(xn,ind3,8),utils::shfl(xn,ind4,8),utils::shfl(xn,ind5,8),
            utils::shfl(yn,ind0,8),utils::shfl(yn,ind1,8),utils::shfl(yn,ind2,8),
            utils::shfl(yn,ind3,8),utils::shfl(yn,ind4,8),utils::shfl(yn,ind5,8),
	          dvdzn);

    /*************************************************/
    /*    compute the hourglass modes                */
    /*************************************************/

    Real_t hourmodx, hourmody, hourmodz;
    const Real_t posf = Real_t( 1.);
    const Real_t negf = Real_t(-1.);

    hourmodx=xn; hourmody=yn; hourmodz=zn;
    if (node==2 || node==3 || node==4 || node==5) {
        hourmodx *= negf; hourmody *= negf; hourmodz *= negf;
        hourgam[0] = negf;
    }
    else hourgam[0] = posf;

    SumOverNodesShfl(hourmodx);
    SumOverNodesShfl(hourmody);
    SumOverNodesShfl(hourmodz);
    hourgam[0] -= volinv*(dvdxn*hourmodx + dvdyn*hourmody + dvdzn*hourmodz);

    
    hourmodx=xn; hourmody=yn; hourmodz=zn;
    if (node==1 || node==2 || node==4 || node==7) {
        hourmodx *= negf; hourmody *= negf; hourmodz *= negf;
        hourgam[1] = negf;
    }
    else hourgam[1] = posf;

    SumOverNodesShfl(hourmodx);
    SumOverNodesShfl(hourmody);
    SumOverNodesShfl(hourmodz);
    hourgam[1] -= volinv*(dvdxn*hourmodx + dvdyn*hourmody + dvdzn*hourmodz);

    
    hourmodx=xn; hourmody=yn; hourmodz=zn;
    if (node==1 || node==3 || node==5 || node==7) {
        hourmodx *= negf; hourmody *= negf; hourmodz *= negf;
        hourgam[2] = negf;
    }
    else hourgam[2] = posf;

    SumOverNodesShfl(hourmodx);
    SumOverNodesShfl(hourmody);
    SumOverNodesShfl(hourmodz);
    hourgam[2] -= volinv*(dvdxn*hourmodx + dvdyn*hourmody + dvdzn*hourmodz);

    
    hourmodx=xn; hourmody=yn; hourmodz=zn;
    if (node==0 || node==2 || node==5 || node==7) {
        hourmodx *= negf; hourmody *= negf; hourmodz *= negf;
        hourgam[3] = negf;
    }
    else hourgam[3] = posf;

    SumOverNodesShfl(hourmodx);
    SumOverNodesShfl(hourmody);
    SumOverNodesShfl(hourmodz);
    hourgam[3] -= volinv*(dvdxn*hourmodx + dvdyn*hourmody + dvdzn*hourmodz);

    /*************************************************/
    /*    CalcStressForElems                         */
    /*************************************************/
    Real_t b[3];

    /*************************************************/
    //CalcElemShapeFunctionDerivatives_warp_per_4cell(xn, yn, zn, B, &det); 
    /*************************************************/

    Real_t fjxxi, fjxet, fjxze;
    Real_t fjyxi, fjyet, fjyze;
    Real_t fjzxi, fjzet, fjzze;

    fjxxi = fjxet = fjxze = Real_t(0.125)*xn;
    fjyxi = fjyet = fjyze = Real_t(0.125)*yn;
    fjzxi = fjzet = fjzze = Real_t(0.125)*zn;

    if (node==0 || node==3 || node==7 || node==4)
    {
      fjxxi = -fjxxi;
      fjyxi = -fjyxi;
      fjzxi = -fjzxi;
    }
    if (node==0 || node==5 || node==1 || node==4)
    {
      fjxet = -fjxet;
      fjyet = -fjyet;
      fjzet = -fjzet;
    }
    if (node==0 || node==3 || node==1 || node==2)
    {
      fjxze = -fjxze;
      fjyze = -fjyze;
      fjzze = -fjzze;
    }

    SumOverNodesShfl(fjxxi); 
    SumOverNodesShfl(fjxet); 
    SumOverNodesShfl(fjxze); 

    SumOverNodesShfl(fjyxi); 
    SumOverNodesShfl(fjyet); 
    SumOverNodesShfl(fjyze); 

    SumOverNodesShfl(fjzxi); 
    SumOverNodesShfl(fjzet); 
    SumOverNodesShfl(fjzze); 

    /* compute cofactors */
    Real_t cjxxi, cjxet, cjxze;
    Real_t cjyxi, cjyet, cjyze;
    Real_t cjzxi, cjzet, cjzze;

    cjxxi =    (fjyet * fjzze) - (fjzet * fjyze);
    cjxet =  - (fjyxi * fjzze) + (fjzxi * fjyze);
    cjxze =    (fjyxi * fjzet) - (fjzxi * fjyet);

    cjyxi =  - (fjxet * fjzze) + (fjzet * fjxze);
    cjyet =    (fjxxi * fjzze) - (fjzxi * fjxze);
    cjyze =  - (fjxxi * fjzet) + (fjzxi * fjxet);

    cjzxi =    (fjxet * fjyze) - (fjyet * fjxze);
    cjzet =  - (fjxxi * fjyze) + (fjyxi * fjxze);
    cjzze =    (fjxxi * fjyet) - (fjyxi * fjxet);

    Real_t coef_xi, coef_et, coef_ze;

    if (node==0 || node==3 || node==4 || node==7)
      coef_xi = Real_t(-1.);
    else
      coef_xi = Real_t(1.);

    if (node==0 || node==1 || node==4 || node==5) 
      coef_et = Real_t(-1.);
    else
      coef_et = Real_t(1.);

    if (node==0 || node==1 || node==2 || node==3) 
      coef_ze = Real_t(-1.);
    else
      coef_ze = Real_t(1.);

    /* calculate partials :
       this need only be done for l = 0,1,2,3   since , by symmetry ,
       (6,7,4,5) = - (0,1,2,3) .
    */
    b[0] =     coef_xi * cjxxi  +  coef_et * cjxet  +  coef_ze * cjxze;
    b[1] =     coef_xi * cjyxi  +  coef_et * cjyet  +  coef_ze * cjyze;
    b[2] =     coef_xi * cjzxi  +  coef_et * cjzet  +  coef_ze * cjzze;

    /* calculate jacobian determinant (volume) */
    det = Real_t(8.) * ( fjxet * cjxet + fjyet * cjyet + fjzet * cjzet);

    /*************************************************/
    //CalcElemNodeNormals_warp_per_4cell( B[0] , B[1], B[2], xn, yn, zn); 
    /*************************************************/

    b[0] = Real_t(0.0);
    b[1] = Real_t(0.0);
    b[2] = Real_t(0.0);

    // Six faces, if no 
    SumElemFaceNormal_warp_per_4cell(&b[0], &b[1], &b[2],
                      xn, yn, zn, node, 0,1,2,3);
                  
    SumElemFaceNormal_warp_per_4cell(&b[0], &b[1], &b[2],
                      xn, yn, zn, node, 0,4,5,1);
 
    SumElemFaceNormal_warp_per_4cell(&b[0], &b[1], &b[2],
                      xn, yn, zn, node, 1,5,6,2);
    
    SumElemFaceNormal_warp_per_4cell(&b[0], &b[1], &b[2],
                      xn, yn, zn, node, 2,6,7,3);

    SumElemFaceNormal_warp_per_4cell(&b[0], &b[1], &b[2],
                      xn, yn, zn, node, 3,7,4,0);

    SumElemFaceNormal_warp_per_4cell(&b[0], &b[1], &b[2],
                      xn, yn, zn, node, 4,7,6,5);

    // Check for bad volume
    if (det < 0.) {
      *bad_vol = elem; 
    }

    hgfx = -( sigxx*b[0] );
    hgfy = -( sigxx*b[1] );
    hgfz = -( sigxx*b[2] );

    if (hourg_gt_zero)
    {
      /*************************************************/
      /*    CalcFBHourglassForceForElems               */
      /*************************************************/

      xdn = xd[node_id];
      ydn = yd[node_id];
      zdn = zd[node_id];

      Real_t hgfx_temp=0;
      #pragma unroll
      for (int i=0;i<4;i++) {
          Real_t h;
          h=hourgam[i]*xdn;
          SumOverNodesShfl(h);
          hgfx_temp+=hourgam[i]*h;
      }
      hgfx_temp *= coefficient;
      hgfx += hgfx_temp;

      Real_t hgfy_temp=0;
      #pragma unroll
      for (int i=0;i<4;i++) {
          Real_t h;
          h=hourgam[i]*ydn;
          SumOverNodesShfl(h);
          hgfy_temp+=hourgam[i]*h;
      }
      hgfy_temp *= coefficient;
      hgfy += hgfy_temp;

      Real_t hgfz_temp=0;
      #pragma unroll
      for (int i=0;i<4;i++) {
          Real_t h;
          h=hourgam[i]*zdn;
          SumOverNodesShfl(h);
          hgfz_temp+=hourgam[i]*h;
      }
      hgfz_temp *= coefficient;
      hgfz += hgfz_temp;


    }

#ifdef DOUBLE_PRECISION
    Index_t store_loc = elem+padded_numElem*node;
    fx_elem[store_loc]=hgfx; 
    fy_elem[store_loc]=hgfy; 
    fz_elem[store_loc]=hgfz;
#else
    atomicAdd(&fx_node[node_id],hgfx); 
    atomicAdd(&fy_node[node_id],hgfy); 
    atomicAdd(&fz_node[node_id],hgfz);
#endif

  } // If elem < numElem
}

static inline
void CalcVolumeForceForElems(const Real_t hgcoef,Domain *domain)
{
    Index_t numElem = domain->numElem ;
    Index_t padded_numElem = domain->padded_numElem;

#ifdef DOUBLE_PRECISION
    Vector_d<Real_t>* fx_elem = Allocator< Vector_d<Real_t> >::allocate(padded_numElem*8);
    Vector_d<Real_t>* fy_elem = Allocator< Vector_d<Real_t> >::allocate(padded_numElem*8);
    Vector_d<Real_t>* fz_elem = Allocator< Vector_d<Real_t> >::allocate(padded_numElem*8);
#else
    thrust::fill(domain->fx.begin(),domain->fx.end(),0.);
    thrust::fill(domain->fy.begin(),domain->fy.end(),0.);
    thrust::fill(domain->fz.begin(),domain->fz.end(),0.);
#endif

    int num_threads = numElem ;
    const int block_size = 64;
    int dimGrid = PAD_DIV(num_threads,block_size);

    bool hourg_gt_zero = hgcoef > Real_t(0.0);
    if (hourg_gt_zero)
    {
      CalcVolumeForceForElems_kernel<true> <<<dimGrid,block_size>>>
      ( domain->volo.raw(), 
        domain->v.raw(), 
        domain->p.raw(), 
        domain->q.raw(),
	      hgcoef, numElem, padded_numElem,
        domain->nodelist.raw(), 
        domain->ss.raw(), 
        domain->elemMass.raw(),
        domain->x.raw(), domain->y.raw(), domain->z.raw(), domain->xd.raw(), domain->yd.raw(), domain->zd.raw(),
#ifdef DOUBLE_PRECISION
        fx_elem->raw(), 
        fy_elem->raw(), 
        fz_elem->raw() ,
#else
        domain->fx.raw(),
        domain->fy.raw(),
        domain->fz.raw(),
#endif
        domain->bad_vol_h,
        num_threads
      );
    }
    else
    {
      CalcVolumeForceForElems_kernel<false> <<<dimGrid,block_size>>>
      ( domain->volo.raw(),
        domain->v.raw(), 
        domain->p.raw(), 
        domain->q.raw(),
	      hgcoef, numElem, padded_numElem,
        domain->nodelist.raw(), 
        domain->ss.raw(), 
        domain->elemMass.raw(),
        domain->x.raw(), domain->y.raw(), domain->z.raw(), domain->xd.raw(), domain->yd.raw(), domain->zd.raw(),
#ifdef DOUBLE_PRECISION
        fx_elem->raw(), 
        fy_elem->raw(), 
        fz_elem->raw() ,
#else
        domain->fx.raw(),
        domain->fy.raw(),
        domain->fz.raw(),
#endif
        domain->bad_vol_h,
        num_threads
      );
    }

#ifdef DOUBLE_PRECISION
    num_threads = domain->numNode;

    // Launch boundary nodes first
    dimGrid= PAD_DIV(num_threads,block_size);

    AddNodeForcesFromElems_kernel<<<dimGrid,block_size>>>
    ( domain->numNode,
      domain->padded_numNode,
      domain->nodeElemCount.raw(),
      domain->nodeElemStart.raw(),
      domain->nodeElemCornerList.raw(),
      fx_elem->raw(),
      fy_elem->raw(),
      fz_elem->raw(),
      domain->fx.raw(),
      domain->fy.raw(),
      domain->fz.raw(),
      num_threads
    );
//    cudaDeviceSynchronize();
//    cudaCheckError();

    Allocator<Vector_d<Real_t> >::free(fx_elem,padded_numElem*8);
    Allocator<Vector_d<Real_t> >::free(fy_elem,padded_numElem*8);
    Allocator<Vector_d<Real_t> >::free(fz_elem,padded_numElem*8);

#endif // ifdef DOUBLE_PRECISION
   return ;
}

/*
static inline
void CalcVolumeForceForElems_warp_per_4cell(const Real_t hgcoef,Domain *domain)
{
  // We're gonna map one warp per 4 cells, i.e. one thread per vertex

    Index_t numElem = domain->numElem ;
    Index_t padded_numElem = domain->padded_numElem;

#ifdef DOUBLE_PRECISION
    Vector_d<Real_t>* fx_elem = Allocator< Vector_d<Real_t> >::allocate(padded_numElem*8);
    Vector_d<Real_t>* fy_elem = Allocator< Vector_d<Real_t> >::allocate(padded_numElem*8);
    Vector_d<Real_t>* fz_elem = Allocator< Vector_d<Real_t> >::allocate(padded_numElem*8);
#else
    thrust::fill(domain->fx.begin(),domain->fx.end(),0.);
    thrust::fill(domain->fy.begin(),domain->fy.end(),0.);
    thrust::fill(domain->fz.begin(),domain->fz.end(),0.);
#endif

    const int warps_per_cta = 2;
    const int cta_size = warps_per_cta * 32;
    int num_threads = numElem*8;

    int dimGrid = PAD_DIV(num_threads,cta_size);

    bool hourg_gt_zero = hgcoef > Real_t(0.0);
    if (hourg_gt_zero)
    {
      CalcVolumeForceForElems_kernel_warp_per_4cell<true, cta_size> <<<dimGrid,cta_size>>>
      ( domain->volo.raw(), 
        domain->v.raw(), 
        domain->p.raw(), 
        domain->q.raw(),
	      hgcoef, numElem, padded_numElem,
        domain->nodelist.raw(), 
        domain->ss.raw(), 
        domain->elemMass.raw(),
        domain->x.raw(), 
        domain->y.raw(), 
        domain->z.raw(), 
        domain->xd.raw(), 
        domain->yd.raw(), 
        domain->zd.raw(), 
        //domain->tex_x, domain->tex_y, domain->tex_z, domain->tex_xd, domain->tex_yd, domain->tex_zd,
#ifdef DOUBLE_PRECISION
        fx_elem->raw(), 
        fy_elem->raw(), 
        fz_elem->raw() ,
#else
        domain->fx.raw(),
        domain->fy.raw(),
        domain->fz.raw(),
#endif
        domain->bad_vol_h,
        num_threads
      );
    }
    else
    {
      CalcVolumeForceForElems_kernel_warp_per_4cell<false, cta_size> <<<dimGrid,cta_size>>>
      ( domain->volo.raw(),
        domain->v.raw(), 
        domain->p.raw(), 
        domain->q.raw(),
	      hgcoef, numElem, padded_numElem,
        domain->nodelist.raw(), 
        domain->ss.raw(), 
        domain->elemMass.raw(),
        domain->x.raw(), 
        domain->y.raw(), 
        domain->z.raw(), 
        domain->xd.raw(), 
        domain->yd.raw(), 
        domain->zd.raw(), 
#ifdef DOUBLE_PRECISION
        fx_elem->raw(), 
        fy_elem->raw(), 
        fz_elem->raw() ,
#else
        domain->fx.raw(),
        domain->fy.raw(),
        domain->fz.raw(),
#endif
        domain->bad_vol_h,
        num_threads
      );
    }

#ifdef DOUBLE_PRECISION
    num_threads = domain->numNode;

    // Launch boundary nodes first
    dimGrid= PAD_DIV(num_threads,cta_size);

    AddNodeForcesFromElems_kernel<<<dimGrid,cta_size>>>
    ( domain->numNode,
      domain->padded_numNode,
      domain->nodeElemCount.raw(),
      domain->nodeElemStart.raw(),
      domain->nodeElemCornerList.raw(),
      fx_elem->raw(),
      fy_elem->raw(),
      fz_elem->raw(),
      domain->fx.raw(),
      domain->fy.raw(),
      domain->fz.raw(),
      num_threads
    );
    //cudaDeviceSynchronize();
    //cudaCheckError();

    Allocator<Vector_d<Real_t> >::free(fx_elem,padded_numElem*8);
    Allocator<Vector_d<Real_t> >::free(fy_elem,padded_numElem*8);
    Allocator<Vector_d<Real_t> >::free(fz_elem,padded_numElem*8);

#endif // ifdef DOUBLE_PRECISION
   return ;
}
*/

static inline
void CalcVolumeForceForElems(Domain* domain)
{
      const Real_t hgcoef = domain->hgcoef ;

     CalcVolumeForceForElems(hgcoef,domain);

     //CalcVolumeForceForElems_warp_per_4cell(hgcoef,domain);
}

static inline void checkErrors(Domain* domain,int its,int myRank)
{
  if (*(domain->bad_vol_h) != -1)
  {
    printf("Rank %i: Volume Error in cell %d at iteration %d\n",myRank,*(domain->bad_vol_h),its);
    exit(VolumeError);
  }

  if (*(domain->bad_q_h) != -1)
  {
    printf("Rank %i: Q Error in cell %d at iteration %d\n",myRank,*(domain->bad_q_h),its);
    exit(QStopError);
  }
}

static inline void CalcForceForNodes(Domain *domain)
{
#if USE_MPI  
  CommRecv(*domain, MSG_COMM_SBN, 3,
           domain->sizeX + 1, domain->sizeY + 1, domain->sizeZ + 1,
           true, false) ;
#endif

  CalcVolumeForceForElems(domain);

  // moved here from the main loop to allow async execution with GPU work
  TimeIncrement(domain);

#if USE_MPI 
  // initialize pointers
  domain->d_fx = domain->fx.raw();
  domain->d_fy = domain->fy.raw();
  domain->d_fz = domain->fz.raw();

  Domain_member fieldData[3] ;
  fieldData[0] = &Domain::get_fx ;
  fieldData[1] = &Domain::get_fy ;
  fieldData[2] = &Domain::get_fz ;

  CommSendGpu(*domain, MSG_COMM_SBN, 3, fieldData,
           domain->sizeX + 1, domain->sizeY + 1, domain->sizeZ + 1,
           true, false, domain->streams[2]) ;
  CommSBNGpu(*domain, 3, fieldData, &domain->streams[2]) ;
#endif
}

__global__
void CalcAccelerationForNodes_kernel(int numNode,
                                     Real_t *xdd, Real_t *ydd, Real_t *zdd,
                                     Real_t *fx, Real_t *fy, Real_t *fz,
                                     Real_t *nodalMass)
{
  int tid=blockDim.x*blockIdx.x+threadIdx.x;
  if (tid < numNode)
  {
      Real_t one_over_nMass = Real_t(1.)/nodalMass[tid];
      xdd[tid]=fx[tid]*one_over_nMass;
      ydd[tid]=fy[tid]*one_over_nMass;
      zdd[tid]=fz[tid]*one_over_nMass;
  }
}

static inline
void CalcAccelerationForNodes(Domain *domain)
{
    Index_t dimBlock = 128;
    Index_t dimGrid = PAD_DIV(domain->numNode,dimBlock);

    CalcAccelerationForNodes_kernel<<<dimGrid, dimBlock>>>
        (domain->numNode,
         domain->xdd.raw(),domain->ydd.raw(),domain->zdd.raw(),
         domain->fx.raw(),domain->fy.raw(),domain->fz.raw(),
         domain->nodalMass.raw());

    //cudaDeviceSynchronize();
    //cudaCheckError();
}

__global__
void ApplyAccelerationBoundaryConditionsForNodes_kernel(
    int numNodeBC, Real_t *xyzdd, 
    Index_t *symm)
{
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if (i < numNodeBC)
    {
          xyzdd[symm[i]] = Real_t(0.0) ;
    }
}

static inline
void ApplyAccelerationBoundaryConditionsForNodes(Domain *domain)
{

    Index_t dimBlock = 128;

    Index_t dimGrid = PAD_DIV(domain->numSymmX,dimBlock);
    if (domain->numSymmX > 0)
      ApplyAccelerationBoundaryConditionsForNodes_kernel<<<dimGrid, dimBlock>>>
        (domain->numSymmX,
         domain->xdd.raw(),
         domain->symmX.raw());

    dimGrid = PAD_DIV(domain->numSymmY,dimBlock);
    if (domain->numSymmY > 0)
      ApplyAccelerationBoundaryConditionsForNodes_kernel<<<dimGrid, dimBlock>>>
        (domain->numSymmY,
         domain->ydd.raw(),
         domain->symmY.raw());

    dimGrid = PAD_DIV(domain->numSymmZ,dimBlock);
    if (domain->numSymmZ > 0)
      ApplyAccelerationBoundaryConditionsForNodes_kernel<<<dimGrid, dimBlock>>>
        (domain->numSymmZ,
         domain->zdd.raw(),
         domain->symmZ.raw());
}


__global__
void CalcPositionAndVelocityForNodes_kernel(int numNode, 
    const Real_t deltatime, 
    const Real_t u_cut,
    Real_t* __restrict__ x,  Real_t* __restrict__ y,  Real_t* __restrict__ z,
    Real_t* __restrict__ xd, Real_t* __restrict__ yd, Real_t* __restrict__ zd,
    const Real_t* __restrict__ xdd, const Real_t* __restrict__ ydd, const Real_t* __restrict__ zdd)
{
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if (i < numNode)
    {
      Real_t xdtmp, ydtmp, zdtmp, dt;
      dt = deltatime;
      
      xdtmp = xd[i] + xdd[i] * dt ;
      ydtmp = yd[i] + ydd[i] * dt ;
      zdtmp = zd[i] + zdd[i] * dt ;

      if( FABS(xdtmp) < u_cut ) xdtmp = 0.0;
      if( FABS(ydtmp) < u_cut ) ydtmp = 0.0;
      if( FABS(zdtmp) < u_cut ) zdtmp = 0.0;

      x[i] += xdtmp * dt;
      y[i] += ydtmp * dt;
      z[i] += zdtmp * dt;
      
      xd[i] = xdtmp; 
      yd[i] = ydtmp; 
      zd[i] = zdtmp; 
    }
}

static inline
void CalcPositionAndVelocityForNodes(const Real_t u_cut, Domain* domain)
{
    Index_t dimBlock = 128;
    Index_t dimGrid = PAD_DIV(domain->numNode,dimBlock);

    CalcPositionAndVelocityForNodes_kernel<<<dimGrid, dimBlock>>>
        (domain->numNode,domain->deltatime_h,u_cut,
         domain->x.raw(),domain->y.raw(),domain->z.raw(),
         domain->xd.raw(),domain->yd.raw(),domain->zd.raw(),
         domain->xdd.raw(),domain->ydd.raw(),domain->zdd.raw());

    //cudaDeviceSynchronize();
    //cudaCheckError();
}

static inline
void LagrangeNodal(Domain *domain)
{
#ifdef SEDOV_SYNC_POS_VEL_EARLY
   Domain_member fieldData[6] ;
#endif

  Real_t u_cut = domain->u_cut ;

  /* time of boundary condition evaluation is beginning of step for force and
   * acceleration boundary conditions. */
  CalcForceForNodes(domain);

#if USE_MPI  
#ifdef SEDOV_SYNC_POS_VEL_EARLY
   CommRecv(*domain, MSG_SYNC_POS_VEL, 6,
            domain->sizeX + 1, domain->sizeY + 1, domain->sizeZ + 1,
            false, false) ;
#endif
#endif

  CalcAccelerationForNodes(domain);

  ApplyAccelerationBoundaryConditionsForNodes(domain);

  CalcPositionAndVelocityForNodes(u_cut, domain);

#if USE_MPI
#ifdef SEDOV_SYNC_POS_VEL_EARLY
  // initialize pointers
  domain->d_x = domain->x.raw();
  domain->d_y = domain->y.raw();
  domain->d_z = domain->z.raw();

  domain->d_xd = domain->xd.raw();
  domain->d_yd = domain->yd.raw();
  domain->d_zd = domain->zd.raw();

  fieldData[0] = &Domain::get_x ;
  fieldData[1] = &Domain::get_y ;
  fieldData[2] = &Domain::get_z ;
  fieldData[3] = &Domain::get_xd ;
  fieldData[4] = &Domain::get_yd ;
  fieldData[5] = &Domain::get_zd ;

  CommSendGpu(*domain, MSG_SYNC_POS_VEL, 6, fieldData,
           domain->sizeX + 1, domain->sizeY + 1, domain->sizeZ + 1,
           false, false, domain->streams[2]) ;
  CommSyncPosVelGpu(*domain, &domain->streams[2]) ;
#endif
#endif

  return;
}

__device__
static inline
Real_t AreaFace( const Real_t x0, const Real_t x1,
                 const Real_t x2, const Real_t x3,
                 const Real_t y0, const Real_t y1,
                 const Real_t y2, const Real_t y3,
                 const Real_t z0, const Real_t z1,
                 const Real_t z2, const Real_t z3)
{
   Real_t fx = (x2 - x0) - (x3 - x1);
   Real_t fy = (y2 - y0) - (y3 - y1);
   Real_t fz = (z2 - z0) - (z3 - z1);
   Real_t gx = (x2 - x0) + (x3 - x1);
   Real_t gy = (y2 - y0) + (y3 - y1);
   Real_t gz = (z2 - z0) + (z3 - z1); 
   Real_t temp = (fx * gx + fy * gy + fz * gz);
   Real_t area =
      (fx * fx + fy * fy + fz * fz) *
      (gx * gx + gy * gy + gz * gz) -
      temp * temp;
   return area ;
}

__device__
static inline
Real_t CalcElemCharacteristicLength( const Real_t x[8],
                                     const Real_t y[8],
                                     const Real_t z[8],
                                     const Real_t volume)
{
   Real_t a, charLength = Real_t(0.0);

   a = AreaFace(x[0],x[1],x[2],x[3],
                y[0],y[1],y[2],y[3],
                z[0],z[1],z[2],z[3]) ; // 38
   charLength = FMAX(a,charLength) ;

   a = AreaFace(x[4],x[5],x[6],x[7],
                y[4],y[5],y[6],y[7],
                z[4],z[5],z[6],z[7]) ;
   charLength = FMAX(a,charLength) ;

   a = AreaFace(x[0],x[1],x[5],x[4],
                y[0],y[1],y[5],y[4],
                z[0],z[1],z[5],z[4]) ;
   charLength = FMAX(a,charLength) ;

   a = AreaFace(x[1],x[2],x[6],x[5],
                y[1],y[2],y[6],y[5],
                z[1],z[2],z[6],z[5]) ;
   charLength = FMAX(a,charLength) ;

   a = AreaFace(x[2],x[3],x[7],x[6],
                y[2],y[3],y[7],y[6],
                z[2],z[3],z[7],z[6]) ;
   charLength = FMAX(a,charLength) ;

   a = AreaFace(x[3],x[0],x[4],x[7],
                y[3],y[0],y[4],y[7],
                z[3],z[0],z[4],z[7]) ;
   charLength = FMAX(a,charLength) ;

   charLength = Real_t(4.0) * volume / SQRT(charLength);

   return charLength;
}

__device__
static 
__forceinline__
void CalcElemVelocityGradient( const Real_t* const xvel,
                                const Real_t* const yvel,
                                const Real_t* const zvel,
                                const Real_t b[][8],
                                const Real_t detJ,
                                Real_t* const d )
{
  const Real_t inv_detJ = Real_t(1.0) / detJ ;
  Real_t dyddx, dxddy, dzddx, dxddz, dzddy, dyddz;
  const Real_t* const pfx = b[0];
  const Real_t* const pfy = b[1];
  const Real_t* const pfz = b[2];
 
  Real_t tmp1 = (xvel[0]-xvel[6]);
  Real_t tmp2 = (xvel[1]-xvel[7]);
  Real_t tmp3 = (xvel[2]-xvel[4]);
  Real_t tmp4 = (xvel[3]-xvel[5]);


  d[0] = inv_detJ *  ( pfx[0] * tmp1
                     + pfx[1] * tmp2
                     + pfx[2] * tmp3
                     + pfx[3] * tmp4);

  dxddy  = inv_detJ * ( pfy[0] * tmp1
                      + pfy[1] * tmp2
                      + pfy[2] * tmp3
                      + pfy[3] * tmp4);

  dxddz  = inv_detJ * ( pfz[0] * tmp1
                      + pfz[1] * tmp2
                      + pfz[2] * tmp3
                      + pfz[3] * tmp4);

  tmp1 = (yvel[0]-yvel[6]);
  tmp2 = (yvel[1]-yvel[7]);
  tmp3 = (yvel[2]-yvel[4]);
  tmp4 = (yvel[3]-yvel[5]);

  d[1] = inv_detJ *  ( pfy[0] * tmp1
                     + pfy[1] * tmp2
                     + pfy[2] * tmp3
                     + pfy[3] * tmp4);

  dyddx  = inv_detJ * ( pfx[0] * tmp1 
                      + pfx[1] * tmp2
                      + pfx[2] * tmp3
                      + pfx[3] * tmp4);

  dyddz  = inv_detJ * ( pfz[0] * tmp1
                      + pfz[1] * tmp2
                      + pfz[2] * tmp3
                      + pfz[3] * tmp4);

  tmp1 = (zvel[0]-zvel[6]);
  tmp2 = (zvel[1]-zvel[7]);
  tmp3 = (zvel[2]-zvel[4]);
  tmp4 = (zvel[3]-zvel[5]);

  d[2] = inv_detJ * ( pfz[0] * tmp1
                     + pfz[1] * tmp2
                     + pfz[2] * tmp3
                     + pfz[3] * tmp4);

  dzddx  = inv_detJ * ( pfx[0] * tmp1 
                      + pfx[1] * tmp2
                      + pfx[2] * tmp3
                      + pfx[3] * tmp4);

  dzddy  = inv_detJ * ( pfy[0] * tmp1
                      + pfy[1] * tmp2
                      + pfy[2] * tmp3
                      + pfy[3] * tmp4);

  d[5]  = Real_t( .5) * ( dxddy + dyddx );
  d[4]  = Real_t( .5) * ( dxddz + dzddx );
  d[3]  = Real_t( .5) * ( dzddy + dyddz );
}

static __device__ __forceinline__
void CalcMonoGradient(Real_t *x, Real_t *y, Real_t *z,
                      Real_t *xv, Real_t *yv, Real_t *zv,
                      Real_t vol, 
                      Real_t *delx_zeta, 
                      Real_t *delv_zeta,
                      Real_t *delx_xi,
                      Real_t *delv_xi,
                      Real_t *delx_eta,
                      Real_t *delv_eta)
{

   #define SUM4(a,b,c,d) (a + b + c + d)
   const Real_t ptiny = Real_t(1.e-36) ;
   Real_t ax,ay,az ;
   Real_t dxv,dyv,dzv ;

   Real_t norm = Real_t(1.0) / ( vol + ptiny ) ;

   Real_t dxj = Real_t(-0.25)*(SUM4(x[0],x[1],x[5],x[4]) - SUM4(x[3],x[2],x[6],x[7])) ;
   Real_t dyj = Real_t(-0.25)*(SUM4(y[0],y[1],y[5],y[4]) - SUM4(y[3],y[2],y[6],y[7])) ;
   Real_t dzj = Real_t(-0.25)*(SUM4(z[0],z[1],z[5],z[4]) - SUM4(z[3],z[2],z[6],z[7])) ;

   Real_t dxi = Real_t( 0.25)*(SUM4(x[1],x[2],x[6],x[5]) - SUM4(x[0],x[3],x[7],x[4])) ;
   Real_t dyi = Real_t( 0.25)*(SUM4(y[1],y[2],y[6],y[5]) - SUM4(y[0],y[3],y[7],y[4])) ;
   Real_t dzi = Real_t( 0.25)*(SUM4(z[1],z[2],z[6],z[5]) - SUM4(z[0],z[3],z[7],z[4])) ;

   Real_t dxk = Real_t( 0.25)*(SUM4(x[4],x[5],x[6],x[7]) - SUM4(x[0],x[1],x[2],x[3])) ;
   Real_t dyk = Real_t( 0.25)*(SUM4(y[4],y[5],y[6],y[7]) - SUM4(y[0],y[1],y[2],y[3])) ;
   Real_t dzk = Real_t( 0.25)*(SUM4(z[4],z[5],z[6],z[7]) - SUM4(z[0],z[1],z[2],z[3])) ;

   /* find delvk and delxk ( i cross j ) */
   ax = dyi*dzj - dzi*dyj ;
   ay = dzi*dxj - dxi*dzj ;
   az = dxi*dyj - dyi*dxj ;

   *delx_zeta = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ; 

   ax *= norm ;
   ay *= norm ;
   az *= norm ; 

   dxv = Real_t(0.25)*(SUM4(xv[4],xv[5],xv[6],xv[7]) - SUM4(xv[0],xv[1],xv[2],xv[3])) ;
   dyv = Real_t(0.25)*(SUM4(yv[4],yv[5],yv[6],yv[7]) - SUM4(yv[0],yv[1],yv[2],yv[3])) ;
   dzv = Real_t(0.25)*(SUM4(zv[4],zv[5],zv[6],zv[7]) - SUM4(zv[0],zv[1],zv[2],zv[3])) ; 

   *delv_zeta = ax*dxv + ay*dyv + az*dzv ;

   /* find delxi and delvi ( j cross k ) */

   ax = dyj*dzk - dzj*dyk ;
   ay = dzj*dxk - dxj*dzk ;
   az = dxj*dyk - dyj*dxk ;

   *delx_xi = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ;

   ax *= norm ;
   ay *= norm ;
   az *= norm ;

   dxv = Real_t(0.25)*(SUM4(xv[1],xv[2],xv[6],xv[5]) - SUM4(xv[0],xv[3],xv[7],xv[4])) ;
   dyv = Real_t(0.25)*(SUM4(yv[1],yv[2],yv[6],yv[5]) - SUM4(yv[0],yv[3],yv[7],yv[4])) ;
   dzv = Real_t(0.25)*(SUM4(zv[1],zv[2],zv[6],zv[5]) - SUM4(zv[0],zv[3],zv[7],zv[4])) ;

   *delv_xi = ax*dxv + ay*dyv + az*dzv ;

   /* find delxj and delvj ( k cross i ) */

   ax = dyk*dzi - dzk*dyi ;
   ay = dzk*dxi - dxk*dzi ;
   az = dxk*dyi - dyk*dxi ;

   *delx_eta = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ;

   ax *= norm ;
   ay *= norm ;
   az *= norm ;

   dxv = Real_t(-0.25)*(SUM4(xv[0],xv[1],xv[5],xv[4]) - SUM4(xv[3],xv[2],xv[6],xv[7])) ;
   dyv = Real_t(-0.25)*(SUM4(yv[0],yv[1],yv[5],yv[4]) - SUM4(yv[3],yv[2],yv[6],yv[7])) ;
   dzv = Real_t(-0.25)*(SUM4(zv[0],zv[1],zv[5],zv[4]) - SUM4(zv[3],zv[2],zv[6],zv[7])) ;

   *delv_eta = ax*dxv + ay*dyv + az*dzv ;
#undef SUM4
}


__global__
#ifdef DOUBLE_PRECISION
__launch_bounds__(64,8) // 64-bit
#else
__launch_bounds__(64,16) // 32-bit
#endif
void CalcKinematicsAndMonotonicQGradient_kernel(
    Index_t numElem, Index_t padded_numElem, const Real_t dt,
    const Index_t* __restrict__ nodelist, const Real_t* __restrict__ volo, const Real_t* __restrict__ v,

    const Real_t* __restrict__ x, 
    const Real_t* __restrict__ y, 
    const Real_t* __restrict__ z,
    const Real_t* __restrict__ xd, 
    const Real_t* __restrict__ yd, 
    const Real_t* __restrict__ zd,

    //TextureObj<Real_t> x, 
    //TextureObj<Real_t> y, 
    //TextureObj<Real_t> z,
    //TextureObj<Real_t> xd, 
    //TextureObj<Real_t> yd, 
    //TextureObj<Real_t> zd,
    //TextureObj<Real_t>* x, 
    //TextureObj<Real_t>* y, 
    //TextureObj<Real_t>* z,
    //TextureObj<Real_t>* xd, 
    //TextureObj<Real_t>* yd, 
    //TextureObj<Real_t>* zd,
 

    Real_t* __restrict__ vnew,
    Real_t* __restrict__ delv,
    Real_t* __restrict__ arealg,
    Real_t* __restrict__ dxx,
    Real_t* __restrict__ dyy,
    Real_t* __restrict__ dzz,
    Real_t* __restrict__ vdov,
    Real_t* __restrict__ delx_zeta, 
    Real_t* __restrict__ delv_zeta,
    Real_t* __restrict__ delx_xi, 
    Real_t* __restrict__ delv_xi, 
    Real_t* __restrict__ delx_eta,
    Real_t* __restrict__ delv_eta,
    Index_t* __restrict__ bad_vol,
    const Index_t num_threads
    )
{

  Real_t B[3][8] ; /** shape function derivatives */
  Index_t nodes[8] ;
  Real_t x_local[8] ;
  Real_t y_local[8] ;
  Real_t z_local[8] ;
  Real_t xd_local[8] ;
  Real_t yd_local[8] ;
  Real_t zd_local[8] ;
  Real_t D[6];

  int k=blockDim.x*blockIdx.x+threadIdx.x;

  if ( k < num_threads) {

    Real_t volume ;
    Real_t relativeVolume ;

    // get nodal coordinates from global arrays and copy into local arrays.
    //#pragma unroll
    //for( Index_t lnode=0 ; lnode<8 ; ++lnode )
    //{
    //  Index_t gnode = nodelist[k+lnode*padded_numElem];
    //  nodes[lnode] = gnode;
    //  x_local[lnode] = x[gnode];
    //  y_local[lnode] = y[gnode];
    //  z_local[lnode] = z[gnode];
    //}

    #pragma unroll
    for( Index_t lnode=0 ; lnode<8 ; ++lnode )
    {
      Index_t gnode = nodelist[k+lnode*padded_numElem];
      nodes[lnode] = gnode;
    }

    #pragma unroll
    for( Index_t lnode=0 ; lnode<8 ; ++lnode )
      x_local[lnode] = x[nodes[lnode]];

    #pragma unroll
    for( Index_t lnode=0 ; lnode<8 ; ++lnode )
      y_local[lnode] = y[nodes[lnode]];

    #pragma unroll
    for( Index_t lnode=0 ; lnode<8 ; ++lnode )
      z_local[lnode] = z[nodes[lnode]];



    // volume calculations
    volume = CalcElemVolume(x_local, y_local, z_local ); 

    relativeVolume = volume / volo[k] ; 
    vnew[k] = relativeVolume ;

    delv[k] = relativeVolume - v[k] ;
    // set characteristic length
    arealg[k] = CalcElemCharacteristicLength(x_local,y_local,z_local,volume);

    // get nodal velocities from global array and copy into local arrays.
    #pragma unroll
    for( Index_t lnode=0 ; lnode<8 ; ++lnode )
    {
      Index_t gnode = nodes[lnode];
      xd_local[lnode] = xd[gnode];
      yd_local[lnode] = yd[gnode];
      zd_local[lnode] = zd[gnode];
    }

    Real_t dt2 = Real_t(0.5) * dt;

    #pragma unroll
    for ( Index_t j=0 ; j<8 ; ++j )
    {
       x_local[j] -= dt2 * xd_local[j];
       y_local[j] -= dt2 * yd_local[j];
       z_local[j] -= dt2 * zd_local[j]; 
    }

    Real_t detJ;

    CalcElemShapeFunctionDerivatives(x_local,y_local,z_local,B,&detJ );

    CalcElemVelocityGradient(xd_local,yd_local,zd_local,B,detJ,D);

    // ------------------------
    // CALC LAGRANGE ELEM 2
    // ------------------------

    // calc strain rate and apply as constraint (only done in FB element)
    Real_t vdovNew = D[0] + D[1] + D[2];
    Real_t vdovthird = vdovNew/Real_t(3.0) ;
    
    // make the rate of deformation tensor deviatoric
    vdov[k] = vdovNew ;
    dxx[k] = D[0] - vdovthird ;
    dyy[k] = D[1] - vdovthird ;
    dzz[k] = D[2] - vdovthird ; 

    // ------------------------
    // CALC MONOTONIC Q GRADIENT
    // ------------------------
    Real_t vol = volo[k]*vnew[k];

   // Undo x_local update
    #pragma unroll
    for ( Index_t j=0 ; j<8 ; ++j ) {
       x_local[j] += dt2 * xd_local[j];
       y_local[j] += dt2 * yd_local[j];
       z_local[j] += dt2 * zd_local[j]; 
    }

   CalcMonoGradient(x_local,y_local,z_local,xd_local,yd_local,zd_local,
                          vol, 
                          &delx_zeta[k],&delv_zeta[k],&delx_xi[k],
                          &delv_xi[k], &delx_eta[k], &delv_eta[k]);

  //Check for bad volume 
  if (relativeVolume < 0)
    *bad_vol = k;
  }
}


static inline
void CalcKinematicsAndMonotonicQGradient(Domain *domain)
{
    Index_t numElem = domain->numElem ;
    Index_t padded_numElem = domain->padded_numElem;

    int num_threads = numElem;

    const int block_size = 64;
    int dimGrid = PAD_DIV(num_threads,block_size);

    CalcKinematicsAndMonotonicQGradient_kernel<<<dimGrid,block_size>>>
    (  numElem,padded_numElem, domain->deltatime_h, 
       domain->nodelist.raw(),
       domain->volo.raw(),
       domain->v.raw(),
       domain->x.raw(), domain->y.raw(), domain->z.raw(), domain->xd.raw(), domain->yd.raw(), domain->zd.raw(),
       domain->vnew->raw(),
       domain->delv.raw(),
       domain->arealg.raw(),
       domain->dxx->raw(),
       domain->dyy->raw(),
       domain->dzz->raw(),
       domain->vdov.raw(), 
       domain->delx_zeta->raw(),
       domain->delv_zeta->raw(), 
       domain->delx_xi->raw(), 
       domain->delv_xi->raw(),  
       domain->delx_eta->raw(), 
       domain->delv_eta->raw(),
       domain->bad_vol_h,
       num_threads  
    );

    //cudaDeviceSynchronize();
    //cudaCheckError();
}

__global__
#ifdef DOUBLE_PRECISION
__launch_bounds__(128,16) 
#else
__launch_bounds__(128,16) 
#endif
void CalcMonotonicQRegionForElems_kernel(
    Real_t qlc_monoq,
    Real_t qqc_monoq,
    Real_t monoq_limiter_mult,
    Real_t monoq_max_slope,
    Real_t ptiny,
    
    // the elementset length
    Index_t elength,
  
    Index_t* regElemlist,  
//    const Index_t* __restrict__ regElemlist,
    Index_t *elemBC,
    Index_t *lxim,
    Index_t *lxip,
    Index_t *letam,
    Index_t *letap,
    Index_t *lzetam,
    Index_t *lzetap,
    Real_t *delv_xi,
    Real_t *delv_eta,
    Real_t *delv_zeta,
    Real_t *delx_xi,
    Real_t *delx_eta,
    Real_t *delx_zeta,
    Real_t *vdov,Real_t *elemMass,Real_t *volo,Real_t *vnew,
    Real_t *qq, Real_t *ql,
    Real_t *q,
    Real_t qstop,
    Index_t* bad_q 
    )
{
    int ielem=blockDim.x*blockIdx.x + threadIdx.x;

    if (ielem<elength) {
      Real_t qlin, qquad ;
      Real_t phixi, phieta, phizeta ;
      Index_t i = regElemlist[ielem];
      Int_t bcMask = elemBC[i] ;
      Real_t delvm, delvp ;

      /*  phixi     */
      Real_t norm = Real_t(1.) / ( delv_xi[i] + ptiny ) ;

      switch (bcMask & XI_M) {
	 case XI_M_COMM: /* needs comm data */
         case 0:         delvm = delv_xi[lxim[i]] ; break ;
         case XI_M_SYMM: delvm = delv_xi[i] ;            break ;
         case XI_M_FREE: delvm = Real_t(0.0) ;                break ;
         default:        /* ERROR */ ;                        break ;
      }
      switch (bcMask & XI_P) {
	 case XI_P_COMM: /* needs comm data */
         case 0:         delvp = delv_xi[lxip[i]] ; break ;
         case XI_P_SYMM: delvp = delv_xi[i] ;            break ;
         case XI_P_FREE: delvp = Real_t(0.0) ;                break ;
         default:        /* ERROR */ ;                        break ;
      }

      delvm = delvm * norm ;
      delvp = delvp * norm ;

      phixi = Real_t(.5) * ( delvm + delvp ) ;

      delvm *= monoq_limiter_mult ;
      delvp *= monoq_limiter_mult ;

      if ( delvm < phixi ) phixi = delvm ;
      if ( delvp < phixi ) phixi = delvp ;
      if ( phixi < Real_t(0.)) phixi = Real_t(0.) ;
      if ( phixi > monoq_max_slope) phixi = monoq_max_slope;


      /*  phieta     */
      norm = Real_t(1.) / ( delv_eta[i] + ptiny ) ;

      switch (bcMask & ETA_M) {
	 case ETA_M_COMM: /* needs comm data */
         case 0:          delvm = delv_eta[letam[i]] ; break ;
         case ETA_M_SYMM: delvm = delv_eta[i] ;             break ;
         case ETA_M_FREE: delvm = Real_t(0.0) ;                  break ;
         default:         /* ERROR */ ;                          break ;
      }
      switch (bcMask & ETA_P) {
	 case ETA_P_COMM: /* needs comm data */
         case 0:          delvp = delv_eta[letap[i]] ; break ;
         case ETA_P_SYMM: delvp = delv_eta[i] ;             break ;
         case ETA_P_FREE: delvp = Real_t(0.0) ;                  break ;
         default:         /* ERROR */ ;                          break ;
      }

      delvm = delvm * norm ;
      delvp = delvp * norm ;

      phieta = Real_t(.5) * ( delvm + delvp ) ;

      delvm *= monoq_limiter_mult ;
      delvp *= monoq_limiter_mult ;

      if ( delvm  < phieta ) phieta = delvm ;
      if ( delvp  < phieta ) phieta = delvp ;
      if ( phieta < Real_t(0.)) phieta = Real_t(0.) ;
      if ( phieta > monoq_max_slope)  phieta = monoq_max_slope;

      /*  phizeta     */
      norm = Real_t(1.) / ( delv_zeta[i] + ptiny ) ;

      switch (bcMask & ZETA_M) {
         case ZETA_M_COMM: /* needs comm data */
         case 0:           delvm = delv_zeta[lzetam[i]] ; break ;
         case ZETA_M_SYMM: delvm = delv_zeta[i] ;              break ;
         case ZETA_M_FREE: delvm = Real_t(0.0) ;                    break ;
         default:          /* ERROR */ ;                            break ;
      }
      switch (bcMask & ZETA_P) {
         case ZETA_P_COMM: /* needs comm data */
         case 0:           delvp = delv_zeta[lzetap[i]] ; break ;
         case ZETA_P_SYMM: delvp = delv_zeta[i] ;              break ;
         case ZETA_P_FREE: delvp = Real_t(0.0) ;                    break ;
         default:          /* ERROR */ ;                            break ;
      }

      delvm = delvm * norm ;
      delvp = delvp * norm ;

      phizeta = Real_t(.5) * ( delvm + delvp ) ;

      delvm *= monoq_limiter_mult ;
      delvp *= monoq_limiter_mult ;

      if ( delvm   < phizeta ) phizeta = delvm ;
      if ( delvp   < phizeta ) phizeta = delvp ;
      if ( phizeta < Real_t(0.)) phizeta = Real_t(0.);
      if ( phizeta > monoq_max_slope  ) phizeta = monoq_max_slope;

      /* Remove length scale */

      if ( vdov[i] > Real_t(0.) ) {
         qlin  = Real_t(0.) ;
         qquad = Real_t(0.) ;
     }
      else {
         Real_t delvxxi   = delv_xi[i]   * delx_xi[i]   ;
         Real_t delvxeta  = delv_eta[i]  * delx_eta[i]  ;
         Real_t delvxzeta = delv_zeta[i] * delx_zeta[i] ;

         if ( delvxxi   > Real_t(0.) ) delvxxi   = Real_t(0.) ;
         if ( delvxeta  > Real_t(0.) ) delvxeta  = Real_t(0.) ;
         if ( delvxzeta > Real_t(0.) ) delvxzeta = Real_t(0.) ;

         Real_t rho = elemMass[i] / (volo[i] * vnew[i]) ;

         qlin = -qlc_monoq * rho *
            (  delvxxi   * (Real_t(1.) - phixi) +
               delvxeta  * (Real_t(1.) - phieta) +
               delvxzeta * (Real_t(1.) - phizeta)  ) ;

         qquad = qqc_monoq * rho *
            (  delvxxi*delvxxi     * (Real_t(1.) - phixi*phixi) +
               delvxeta*delvxeta   * (Real_t(1.) - phieta*phieta) +
               delvxzeta*delvxzeta * (Real_t(1.) - phizeta*phizeta)  ) ;
      }

      qq[i] = qquad ;
      ql[i] = qlin  ;

      // Don't allow excessive artificial viscosity
      if (q[i] > qstop)
        *(bad_q) = i;

   }
}


static inline
void CalcMonotonicQRegionForElems(Domain *domain)
{

    const Real_t ptiny        = Real_t(1.e-36) ;
    Real_t monoq_max_slope    = domain->monoq_max_slope ;
    Real_t monoq_limiter_mult = domain->monoq_limiter_mult ;

    Real_t qlc_monoq = domain->qlc_monoq;
    Real_t qqc_monoq = domain->qqc_monoq;
    Index_t elength = domain->numElem;

    Index_t dimBlock= 128;
    Index_t dimGrid = PAD_DIV(elength,dimBlock);

    CalcMonotonicQRegionForElems_kernel<<<dimGrid,dimBlock>>>
    ( qlc_monoq,qqc_monoq,monoq_limiter_mult,monoq_max_slope,ptiny,elength,
      domain->regElemlist.raw(),domain->elemBC.raw(),
      domain->lxim.raw(),domain->lxip.raw(),
      domain->letam.raw(),domain->letap.raw(),
      domain->lzetam.raw(),domain->lzetap.raw(),
      domain->delv_xi->raw(),domain->delv_eta->raw(),domain->delv_zeta->raw(),
      domain->delx_xi->raw(),domain->delx_eta->raw(),domain->delx_zeta->raw(),
      domain->vdov.raw(),domain->elemMass.raw(),domain->volo.raw(),domain->vnew->raw(),
      domain->qq.raw(),domain->ql.raw(), 
      domain->q.raw(),
      domain->qstop,
      domain->bad_q_h
    );

    //cudaDeviceSynchronize();
    //cudaCheckError();
}

static 
__device__ __forceinline__
void CalcPressureForElems_device(
                      Real_t& p_new, Real_t& bvc,
                      Real_t& pbvc, Real_t& e_old,
                      Real_t& compression, Real_t& vnewc,
                      Real_t pmin,
                      Real_t p_cut, Real_t eosvmax)
{
      
      Real_t c1s = Real_t(2.0)/Real_t(3.0); 
      Real_t p_temp = p_new;

      bvc = c1s * (compression + Real_t(1.));
      pbvc = c1s;

      p_temp = bvc * e_old ;

      if ( FABS(p_temp) <  p_cut )
        p_temp = Real_t(0.0) ;

      if ( vnewc >= eosvmax ) /* impossible condition here? */
        p_temp = Real_t(0.0) ;

      if (p_temp < pmin)
        p_temp = pmin ;

      p_new = p_temp;

}

static
__device__ __forceinline__
void CalcSoundSpeedForElems_device(Real_t& vnewc, Real_t rho0, Real_t &enewc,
                            Real_t &pnewc, Real_t &pbvc,
                            Real_t &bvc, Real_t ss4o3, Index_t nz,
                            Real_t *ss, Index_t iz)
{
  Real_t ssTmp = (pbvc * enewc + vnewc * vnewc *
             bvc * pnewc) / rho0;
  if (ssTmp <= Real_t(.1111111e-36)) {
     ssTmp = Real_t(.3333333e-18);
  }
  else {
    ssTmp = SQRT(ssTmp) ;
  }
  ss[iz] = ssTmp;
}

static
__device__
__forceinline__ 
void ApplyMaterialPropertiesForElems_device(
    Real_t& eosvmin, Real_t& eosvmax,
    Real_t* vnew, Real_t *v,
    Real_t& vnewc, Index_t* bad_vol, Index_t zn)
{
  vnewc = vnew[zn] ;

  if (eosvmin != Real_t(0.)) {
      if (vnewc < eosvmin)
          vnewc = eosvmin ;
  }

  if (eosvmax != Real_t(0.)) {
      if (vnewc > eosvmax)
          vnewc = eosvmax ;
  }

  // Now check for valid volume
  Real_t vc = v[zn];
  if (eosvmin != Real_t(0.)) {
     if (vc < eosvmin)
        vc = eosvmin ;
  }
  if (eosvmax != Real_t(0.)) {
     if (vc > eosvmax)
        vc = eosvmax ;
  }
  if (vc <= 0.) {
     *bad_vol = zn;
  }

}

static
__device__
__forceinline__
void UpdateVolumesForElems_device(Index_t numElem, Real_t& v_cut,
                                  Real_t *vnew,
                                  Real_t *v,
                                  int i)
{
   Real_t tmpV ;
   tmpV = vnew[i] ;

   if ( FABS(tmpV - Real_t(1.0)) < v_cut )
      tmpV = Real_t(1.0) ;
   v[i] = tmpV ;
}


static
__device__
__forceinline__
void CalcEnergyForElems_device(Real_t& p_new, Real_t& e_new, Real_t& q_new,
                            Real_t& bvc, Real_t& pbvc,
                            Real_t& p_old, Real_t& e_old, Real_t& q_old,
                            Real_t& compression, Real_t& compHalfStep,
                            Real_t& vnewc, Real_t& work, Real_t& delvc, Real_t pmin,
                            Real_t p_cut, Real_t e_cut, Real_t q_cut, Real_t emin,
                            Real_t& qq, Real_t& ql,
                            Real_t& rho0,
                            Real_t& eosvmax,
                            Index_t length)
{
   const Real_t sixth = Real_t(1.0) / Real_t(6.0) ;
   Real_t pHalfStep;

   e_new = e_old - Real_t(0.5) * delvc * (p_old + q_old)
      + Real_t(0.5) * work;

   if (e_new  < emin ) {
      e_new = emin ;
   }

   CalcPressureForElems_device(pHalfStep, bvc, pbvc, e_new, compHalfStep, vnewc,
                   pmin, p_cut, eosvmax);

   Real_t vhalf = Real_t(1.) / (Real_t(1.) + compHalfStep) ;

   if ( delvc > Real_t(0.) ) {
      q_new = Real_t(0.) ;
   }
   else {
      Real_t ssc = ( pbvc * e_new
              + vhalf * vhalf * bvc * pHalfStep ) / rho0 ;

      if ( ssc <= Real_t(.1111111e-36) ) {
         ssc =Real_t(.3333333e-18) ;
      } else {
         ssc = SQRT(ssc) ;
      }

      q_new = (ssc*ql + qq) ;
   }

   e_new = e_new + Real_t(0.5) * delvc
      * (  Real_t(3.0)*(p_old     + q_old)
           - Real_t(4.0)*(pHalfStep + q_new)) ;

   e_new += Real_t(0.5) * work;

   if (FABS(e_new) < e_cut) {
      e_new = Real_t(0.)  ;
   }
   if (     e_new  < emin ) {
      e_new = emin ;
   }

   CalcPressureForElems_device(p_new, bvc, pbvc, e_new, compression, vnewc,
                   pmin, p_cut, eosvmax);

   Real_t q_tilde ;

   if (delvc > Real_t(0.)) {
      q_tilde = Real_t(0.) ;
   }
   else {
      Real_t ssc = ( pbvc * e_new
              + vnewc * vnewc * bvc * p_new ) / rho0 ;

      if ( ssc <= Real_t(.1111111e-36) ) {
         ssc = Real_t(.3333333e-18) ;
      } else {
         ssc = SQRT(ssc) ;
      }

      q_tilde = (ssc*ql + qq) ;
   }

   e_new = e_new - (  Real_t(7.0)*(p_old     + q_old)
                            - Real_t(8.0)*(pHalfStep + q_new)
                            + (p_new + q_tilde)) * delvc*sixth ;

   if (FABS(e_new) < e_cut) {
      e_new = Real_t(0.)  ;
   }
   if ( e_new  < emin ) {
      e_new = emin ;
   }

   CalcPressureForElems_device(p_new, bvc, pbvc, e_new, compression, vnewc,
                   pmin, p_cut, eosvmax);


   if ( delvc <= Real_t(0.) ) {
      Real_t ssc = ( pbvc * e_new
              + vnewc * vnewc * bvc * p_new ) / rho0 ;

      if ( ssc <= Real_t(.1111111e-36) ) {
         ssc = Real_t(.3333333e-18) ;
      } else {
         ssc = SQRT(ssc) ;
      }

      q_new = (ssc*ql + qq) ;

      if (FABS(q_new) < q_cut) q_new = Real_t(0.) ;
   }

   return ;
}

__device__ inline
Index_t giveMyRegion(const Index_t* regCSR,const Index_t i, const Index_t numReg)
{

  for(Index_t reg = 0; reg < numReg-1; reg++)
	if(i < regCSR[reg])
		return reg;
  return (numReg-1);
}


__global__
void ApplyMaterialPropertiesAndUpdateVolume_kernel(
        Index_t length,
        Real_t rho0,
        Real_t e_cut,
        Real_t emin,
        Real_t* __restrict__ ql,
        Real_t* __restrict__ qq,
        Real_t* __restrict__ vnew,
        Real_t* __restrict__ v,
        Real_t pmin,
        Real_t p_cut,
        Real_t q_cut,
        Real_t eosvmin,
        Real_t eosvmax,
        Index_t* __restrict__ regElemlist,
//        const Index_t* __restrict__ regElemlist,
        Real_t* __restrict__ e,
        Real_t* __restrict__ delv,
        Real_t* __restrict__ p,
        Real_t* __restrict__ q,
        Real_t ss4o3,
        Real_t* __restrict__ ss,
        Real_t v_cut,
        Index_t* __restrict__ bad_vol, 
        const Int_t cost,
        const Index_t* regCSR,
        const Index_t* regReps,
	const Index_t  numReg
)
{

  Real_t e_old, delvc, p_old, q_old, e_temp, delvc_temp, p_temp, q_temp;
  Real_t compression, compHalfStep;
  Real_t qq_old, ql_old, qq_temp, ql_temp, work;
  Real_t p_new, e_new, q_new;
  Real_t bvc, pbvc, vnewc;

  Index_t i=blockDim.x*blockIdx.x + threadIdx.x;

  if (i<length) {

    Index_t zidx  = regElemlist[i] ;

    ApplyMaterialPropertiesForElems_device
      (eosvmin,eosvmax,vnew,v,vnewc,bad_vol,zidx);
/********************** Start EvalEOSForElems   **************************/
// Here we need to find out what region this element belongs to and what is the rep value!
  Index_t region = giveMyRegion(regCSR,i,numReg);  
  Index_t rep = regReps[region];
    
    e_temp = e[zidx];
    p_temp = p[zidx];
    q_temp = q[zidx];
    qq_temp = qq[zidx] ;
    ql_temp = ql[zidx] ;
    delvc_temp = delv[zidx];

  for(int r=0; r < rep; r++)
  {

    e_old = e_temp;
    p_old = p_temp;
    q_old = q_temp;
    qq_old = qq_temp;
    ql_old = ql_temp;
    delvc = delvc_temp;
    work = Real_t(0.);

    Real_t vchalf ;
    compression = Real_t(1.) / vnewc - Real_t(1.);
    vchalf = vnewc - delvc * Real_t(.5);
    compHalfStep = Real_t(1.) / vchalf - Real_t(1.);

    if ( eosvmin != Real_t(0.) ) {
        if (vnewc <= eosvmin) { /* impossible due to calling func? */
            compHalfStep = compression ;
        }
    }
    if ( eosvmax != Real_t(0.) ) {
        if (vnewc >= eosvmax) { /* impossible due to calling func? */
            p_old        = Real_t(0.) ;
            compression  = Real_t(0.) ;
            compHalfStep = Real_t(0.) ;
        }
    }

//    qq_old = qq[zidx] ;
//    ql_old = ql[zidx] ;
//    work = Real_t(0.) ;

    CalcEnergyForElems_device(p_new, e_new, q_new, bvc, pbvc,
                 p_old, e_old,  q_old, compression, compHalfStep,
                 vnewc, work,  delvc, pmin,
                 p_cut, e_cut, q_cut, emin,
                 qq_old, ql_old, rho0, eosvmax, length);
 }//end for rep

    p[zidx] = p_new ;
    e[zidx] = e_new ;
    q[zidx] = q_new ;

    CalcSoundSpeedForElems_device
       (vnewc,rho0,e_new,p_new,pbvc,bvc,ss4o3,length,ss,zidx);

/********************** End EvalEOSForElems     **************************/

    UpdateVolumesForElems_device(length,v_cut,vnew,v,zidx);

  }
}

static inline
void ApplyMaterialPropertiesAndUpdateVolume(Domain *domain)
{
  Index_t length = domain->numElem ;

  if (length != 0) {

    Index_t dimBlock = 128;
    Index_t dimGrid = PAD_DIV(length,dimBlock);

    ApplyMaterialPropertiesAndUpdateVolume_kernel<<<dimGrid,dimBlock>>>
        (length,
         domain->refdens,
         domain->e_cut,
         domain->emin,
         domain->ql.raw(),
         domain->qq.raw(),
         domain->vnew->raw(),
         domain->v.raw(),
         domain->pmin,
         domain->p_cut,
         domain->q_cut,
         domain->eosvmin,
         domain->eosvmax,
         domain->regElemlist.raw(),
         domain->e.raw(),
         domain->delv.raw(),
         domain->p.raw(),
         domain->q.raw(),
         domain->ss4o3,
         domain->ss.raw(),
         domain->v_cut,
         domain->bad_vol_h,
	 domain->cost,
	 domain->regCSR.raw(),
	 domain->regReps.raw(),
	 domain->numReg
         );

    //cudaDeviceSynchronize();
    //cudaCheckError();
  }
}

static inline
void LagrangeElements(Domain *domain)
{
  int allElem = domain->numElem +  /* local elem */
                2*domain->sizeX*domain->sizeY + /* plane ghosts */
                2*domain->sizeX*domain->sizeZ + /* row ghosts */
                2*domain->sizeY*domain->sizeZ ; /* col ghosts */

  domain->vnew = Allocator< Vector_d<Real_t> >::allocate(domain->numElem);
  domain->dxx  = Allocator< Vector_d<Real_t> >::allocate(domain->numElem);
  domain->dyy  = Allocator< Vector_d<Real_t> >::allocate(domain->numElem);
  domain->dzz  = Allocator< Vector_d<Real_t> >::allocate(domain->numElem);

  domain->delx_xi    = Allocator< Vector_d<Real_t> >::allocate(domain->numElem);
  domain->delx_eta   = Allocator< Vector_d<Real_t> >::allocate(domain->numElem);
  domain->delx_zeta  = Allocator< Vector_d<Real_t> >::allocate(domain->numElem);

  domain->delv_xi    = Allocator< Vector_d<Real_t> >::allocate(allElem);
  domain->delv_eta   = Allocator< Vector_d<Real_t> >::allocate(allElem);
  domain->delv_zeta  = Allocator< Vector_d<Real_t> >::allocate(allElem);

#if USE_MPI     
  CommRecv(*domain, MSG_MONOQ, 3,
           domain->sizeX, domain->sizeY, domain->sizeZ,
           true, true) ;
#endif

  /*********************************************/
  /*  Calc Kinematics and Monotic Q Gradient   */
  /*********************************************/
  CalcKinematicsAndMonotonicQGradient(domain);

#if USE_MPI      
   Domain_member fieldData[3] ;

   // initialize pointers
   domain->d_delv_xi = domain->delv_xi->raw();
   domain->d_delv_eta = domain->delv_eta->raw();
   domain->d_delv_zeta = domain->delv_zeta->raw();

   fieldData[0] = &Domain::get_delv_xi ;
   fieldData[1] = &Domain::get_delv_eta ;
   fieldData[2] = &Domain::get_delv_zeta ;

   CommSendGpu(*domain, MSG_MONOQ, 3, fieldData,
            domain->sizeX, domain->sizeY, domain->sizeZ,
            true, true, domain->streams[2]) ;
   CommMonoQGpu(*domain, domain->streams[2]) ;
#endif

  Allocator<Vector_d<Real_t> >::free(domain->dxx,domain->numElem);
  Allocator<Vector_d<Real_t> >::free(domain->dyy,domain->numElem);
  Allocator<Vector_d<Real_t> >::free(domain->dzz,domain->numElem);

  /**********************************
  *    Calc Monotic Q Region
  **********************************/
   CalcMonotonicQRegionForElems(domain);

  Allocator<Vector_d<Real_t> >::free(domain->delx_xi,domain->numElem);
  Allocator<Vector_d<Real_t> >::free(domain->delx_eta,domain->numElem);
  Allocator<Vector_d<Real_t> >::free(domain->delx_zeta,domain->numElem);

  Allocator<Vector_d<Real_t> >::free(domain->delv_xi,allElem);
  Allocator<Vector_d<Real_t> >::free(domain->delv_eta,allElem);
  Allocator<Vector_d<Real_t> >::free(domain->delv_zeta,allElem);

//  printf("\n --Start of ApplyMaterials! \n");
  ApplyMaterialPropertiesAndUpdateVolume(domain) ;
//  printf("\n --End of ApplyMaterials! \n");
  Allocator<Vector_d<Real_t> >::free(domain->vnew,domain->numElem);
}

template<int block_size>
__global__
#ifdef DOUBLE_PRECISION
__launch_bounds__(128,16) 
#else
__launch_bounds__(128,16) 
#endif
void CalcTimeConstraintsForElems_kernel(
    Index_t length,
    Real_t qqc2, 
    Real_t dvovmax,
    Index_t *matElemlist,
    Real_t *ss,
    Real_t *vdov,
    Real_t *arealg,
    Real_t *dev_mindtcourant,
    Real_t *dev_mindthydro)
{
    int tid = threadIdx.x;
    int i=blockDim.x*blockIdx.x + tid;

    __shared__ volatile Real_t s_mindthydro[block_size];
    __shared__ volatile Real_t s_mindtcourant[block_size];

    Real_t mindthydro = Real_t(1.0e+20) ;
    Real_t mindtcourant = Real_t(1.0e+20) ;

    Real_t dthydro = mindthydro;
    Real_t dtcourant = mindtcourant;

    while (i<length) {

      Index_t indx = matElemlist[i] ;
      Real_t vdov_tmp = vdov[indx];

      // Computing dt_hydro
      if (vdov_tmp != Real_t(0.)) {
         Real_t dtdvov = dvovmax / (FABS(vdov_tmp)+Real_t(1.e-20)) ;
         if ( dthydro > dtdvov ) {
            dthydro = dtdvov ;
         }
      }
      if (dthydro < mindthydro)
        mindthydro = dthydro;

      // Computing dt_courant
      Real_t ss_tmp = ss[indx];
      Real_t area_tmp = arealg[indx];
      Real_t dtf = ss_tmp * ss_tmp ;
    
      dtf += ((vdov_tmp < 0.) ? qqc2*area_tmp*area_tmp*vdov_tmp*vdov_tmp : 0.);

      dtf = area_tmp / SQRT(dtf) ;

      /* determine minimum timestep with its corresponding elem */
      if (vdov_tmp != Real_t(0.) && dtf < dtcourant) {
        dtcourant = dtf ;
      }

      if (dtcourant< mindtcourant)
        mindtcourant= dtcourant;

      i += gridDim.x*blockDim.x;
    }

    s_mindthydro[tid]   = mindthydro;
    s_mindtcourant[tid] = mindtcourant;

    __syncthreads();

    // Do shared memory reduction
    if (block_size >= 1024) { 
      if (tid < 512) { 
        s_mindthydro[tid]   = min( s_mindthydro[tid]  , s_mindthydro[tid + 512]) ; 
        s_mindtcourant[tid] = min( s_mindtcourant[tid], s_mindtcourant[tid + 512]) ; }
      __syncthreads();  }

    if (block_size >=  512) { 
      if (tid < 256) { 
        s_mindthydro[tid] = min( s_mindthydro[tid], s_mindthydro[tid + 256]) ; 
        s_mindtcourant[tid] = min( s_mindtcourant[tid], s_mindtcourant[tid + 256]) ; } 
      __syncthreads(); }

    if (block_size >=  256) { 
      if (tid < 128) { 
        s_mindthydro[tid] = min( s_mindthydro[tid], s_mindthydro[tid + 128]) ; 
        s_mindtcourant[tid] = min( s_mindtcourant[tid], s_mindtcourant[tid + 128]) ; } 
      __syncthreads(); }

    if (block_size >=  128) { 
      if (tid <  64) { 
        s_mindthydro[tid] = min( s_mindthydro[tid], s_mindthydro[tid +  64]) ; 
        s_mindtcourant[tid] = min( s_mindtcourant[tid], s_mindtcourant[tid +  64]) ; } 
      __syncthreads(); }

    if (tid <  32) { 
      s_mindthydro[tid] = min( s_mindthydro[tid], s_mindthydro[tid +  32]) ; 
      s_mindtcourant[tid] = min( s_mindtcourant[tid], s_mindtcourant[tid +  32]) ; 
    } 

    if (tid <  16) { 
      s_mindthydro[tid] = min( s_mindthydro[tid], s_mindthydro[tid +  16]) ; 
      s_mindtcourant[tid] = min( s_mindtcourant[tid], s_mindtcourant[tid +  16]) ;  
    } 
    if (tid <   8) { 
      s_mindthydro[tid] = min( s_mindthydro[tid], s_mindthydro[tid +   8]) ; 
      s_mindtcourant[tid] = min( s_mindtcourant[tid], s_mindtcourant[tid +   8]) ;  
    } 
    if (tid <   4) { 
      s_mindthydro[tid] = min( s_mindthydro[tid], s_mindthydro[tid +   4]) ; 
      s_mindtcourant[tid] = min( s_mindtcourant[tid], s_mindtcourant[tid +   4]) ;  
    } 
    if (tid <   2) { 
      s_mindthydro[tid] = min( s_mindthydro[tid], s_mindthydro[tid +   2]) ; 
      s_mindtcourant[tid] = min( s_mindtcourant[tid], s_mindtcourant[tid +   2]) ;  
    } 
    if (tid <   1) { 
      s_mindthydro[tid] = min( s_mindthydro[tid], s_mindthydro[tid +   1]) ; 
      s_mindtcourant[tid] = min( s_mindtcourant[tid], s_mindtcourant[tid +   1]) ;  
    } 

    // Store in global memory
    if (tid==0) {
      dev_mindtcourant[blockIdx.x] = s_mindtcourant[0];
      dev_mindthydro[blockIdx.x] = s_mindthydro[0];
    }

}

template <int block_size>
__global__ 
void CalcMinDtOneBlock(Real_t* dev_mindthydro, Real_t* dev_mindtcourant, Real_t* dtcourant, Real_t* dthydro, Index_t shared_array_size)
{

  volatile __shared__ Real_t s_data[block_size];
  int tid = threadIdx.x;

  if (blockIdx.x==0)
  {
    if (tid < shared_array_size)
      s_data[tid] = dev_mindtcourant[tid];
    else
      s_data[tid] = 1.0e20;

    __syncthreads();

    if (block_size >= 1024) { if (tid < 512) { s_data[tid] = min(s_data[tid],s_data[tid + 512]); } __syncthreads(); }
    if (block_size >=  512) { if (tid < 256) { s_data[tid] = min(s_data[tid],s_data[tid + 256]); } __syncthreads(); }
    if (block_size >=  256) { if (tid < 128) { s_data[tid] = min(s_data[tid],s_data[tid + 128]); } __syncthreads(); }
    if (block_size >=  128) { if (tid <  64) { s_data[tid] = min(s_data[tid],s_data[tid +  64]); } __syncthreads(); }
    if (tid <  32) { s_data[tid] = min(s_data[tid],s_data[tid +  32]); } 
    if (tid <  16) { s_data[tid] = min(s_data[tid],s_data[tid +  16]); } 
    if (tid <   8) { s_data[tid] = min(s_data[tid],s_data[tid +   8]); } 
    if (tid <   4) { s_data[tid] = min(s_data[tid],s_data[tid +   4]); } 
    if (tid <   2) { s_data[tid] = min(s_data[tid],s_data[tid +   2]); } 
    if (tid <   1) { s_data[tid] = min(s_data[tid],s_data[tid +   1]); } 

    if (tid<1)
    {
      *(dtcourant)= s_data[0];
    }
  }
  else if (blockIdx.x==1)
  {
    if (tid < shared_array_size)
      s_data[tid] = dev_mindthydro[tid];
    else
      s_data[tid] = 1.0e20;

    __syncthreads();

    if (block_size >= 1024) { if (tid < 512) { s_data[tid] = min(s_data[tid],s_data[tid + 512]); } __syncthreads(); }
    if (block_size >=  512) { if (tid < 256) { s_data[tid] = min(s_data[tid],s_data[tid + 256]); } __syncthreads(); }
    if (block_size >=  256) { if (tid < 128) { s_data[tid] = min(s_data[tid],s_data[tid + 128]); } __syncthreads(); }
    if (block_size >=  128) { if (tid <  64) { s_data[tid] = min(s_data[tid],s_data[tid +  64]); } __syncthreads(); }
    if (tid <  32) { s_data[tid] = min(s_data[tid],s_data[tid +  32]); } 
    if (tid <  16) { s_data[tid] = min(s_data[tid],s_data[tid +  16]); } 
    if (tid <   8) { s_data[tid] = min(s_data[tid],s_data[tid +   8]); } 
    if (tid <   4) { s_data[tid] = min(s_data[tid],s_data[tid +   4]); } 
    if (tid <   2) { s_data[tid] = min(s_data[tid],s_data[tid +   2]); } 
    if (tid <   1) { s_data[tid] = min(s_data[tid],s_data[tid +   1]); } 

    if (tid<1)
    {
      *(dthydro) = s_data[0];
    }
  }
}

static inline
void CalcTimeConstraintsForElems(Domain* domain)
{
    Real_t qqc = domain->qqc;
    Real_t qqc2 = Real_t(64.0) * qqc * qqc ;
    Real_t dvovmax = domain->dvovmax ;

    const Index_t length = domain->numElem;

    const int max_dimGrid = 1024;
    const int dimBlock = 128;
    int dimGrid=std::min(max_dimGrid,PAD_DIV(length,dimBlock));

    cudaFuncSetCacheConfig(CalcTimeConstraintsForElems_kernel<dimBlock>, cudaFuncCachePreferShared);

    Vector_d<Real_t>* dev_mindtcourant= Allocator< Vector_d<Real_t> >::allocate(dimGrid);
    Vector_d<Real_t>* dev_mindthydro  = Allocator< Vector_d<Real_t> >::allocate(dimGrid);

    CalcTimeConstraintsForElems_kernel<dimBlock> <<<dimGrid,dimBlock>>>
        (length,qqc2,dvovmax,
         domain->matElemlist.raw(),domain->ss.raw(),domain->vdov.raw(),domain->arealg.raw(),
         dev_mindtcourant->raw(),dev_mindthydro->raw());

    // TODO: if dimGrid < 1024, should launch less threads
    CalcMinDtOneBlock<max_dimGrid> <<<2,max_dimGrid, max_dimGrid*sizeof(Real_t), domain->streams[1]>>>(dev_mindthydro->raw(),dev_mindtcourant->raw(),domain->dtcourant_h,domain->dthydro_h, dimGrid);

    cudaEventRecord(domain->time_constraint_computed,domain->streams[1]);

    Allocator<Vector_d<Real_t> >::free(dev_mindtcourant,dimGrid);
    Allocator<Vector_d<Real_t> >::free(dev_mindthydro,dimGrid);
}


static inline
void LagrangeLeapFrog(Domain* domain)
{

   /* calculate nodal forces, accelerations, velocities, positions, with
    * applied boundary conditions and slide surface considerations */
   LagrangeNodal(domain);

   /* calculate element quantities (i.e. velocity gradient & q), and update
    * material states */
   LagrangeElements(domain);

   CalcTimeConstraintsForElems(domain);

}

void printUsage(char* argv[])
{
  printf("Usage: \n");
  printf("Unstructured grid:  %s -u <file.lmesh> \n", argv[0]) ;
  printf("Structured grid:    %s -s numEdgeElems \n", argv[0]) ;
  printf("\nExamples:\n") ;
  printf("%s -s 45\n", argv[0]) ;
  printf("%s -u sedov15oct.lmesh\n", argv[0]) ;
}

#ifdef SAMI

#ifdef __cplusplus
  extern "C" {
#endif
#include "silo.h"
#ifdef __cplusplus
  }
#endif

#define MAX_LEN_SAMI_HEADER  10

#define SAMI_HDR_NUMBRICK     0
#define SAMI_HDR_NUMNODES     3
#define SAMI_HDR_NUMMATERIAL  4
#define SAMI_HDR_INDEX_START  6
#define SAMI_HDR_MESHDIM      7

#define MAX_ADJACENCY  14  /* must be 14 or greater */

void DumpSAMI(Domain *domain, char *name)
{
   DBfile *fp ;
   int headerLen = MAX_LEN_SAMI_HEADER ;
   int headerInfo[MAX_LEN_SAMI_HEADER];
   char varName[] = "brick_nd0";
   char coordName[] = "x";
   int version = 121 ;
   int numElem = int(domain->numElem) ;
   int numNode = int(domain->numNode) ;
   int count ;

   int *materialID ;
   int *nodeConnect ;
   double *nodeCoord ;

   if ((fp = DBCreate(name, DB_CLOBBER, DB_LOCAL,
                        NULL, DB_PDB)) == NULL)
   {
      printf("Couldn't create file %s\n", name) ;
      exit(1);
   }

   for (int i=0; i<MAX_LEN_SAMI_HEADER; ++i) {
      headerInfo[i] = 0 ;
   }
   headerInfo[SAMI_HDR_NUMBRICK]    = numElem ;
   headerInfo[SAMI_HDR_NUMNODES]    = numNode ;
   headerInfo[SAMI_HDR_NUMMATERIAL] = 1 ;
   headerInfo[SAMI_HDR_INDEX_START] = 1 ;
   headerInfo[SAMI_HDR_MESHDIM]     = 3 ;

   DBWrite(fp, "mesh_data", headerInfo, &headerLen, 1, DB_INT) ;

   count = 1 ;
   DBWrite(fp, "version", &version, &count, 1, DB_INT) ;

   nodeConnect = new int[numElem] ;

   Vector_h<Index_t> nodelist_h = domain->nodelist;

   for (Index_t i=0; i<8; ++i)
   {
      for (Index_t j=0; j<numElem; ++j) {
         nodeConnect[j] = int(nodelist_h[i*domain->padded_numElem + j]) + 1 ;
      }
      varName[8] = '0' + i;
      DBWrite(fp, varName, nodeConnect, &numElem, 1, DB_INT) ;
   }

   delete [] nodeConnect ;

   nodeCoord = new double[numNode] ;

   Vector_h<Real_t> x_h = domain->x;
   Vector_h<Real_t> y_h = domain->y;
   Vector_h<Real_t> z_h = domain->z;

   for (Index_t i=0; i<3; ++i)
   {
      for (Index_t j=0; j<numNode; ++j) {
         Real_t coordVal ;
         switch(i) {
            case 0: coordVal = double(x_h[j]) ; break ;
            case 1: coordVal = double(y_h[j]) ; break ;
            case 2: coordVal = double(z_h[j]) ; break ;
         }
         nodeCoord[j] = coordVal ;
      }
      coordName[0] = 'x' + i ;
      DBWrite(fp, coordName, nodeCoord, &numNode, 1, DB_DOUBLE) ;
   }

   delete [] nodeCoord ;

   materialID = new int[numElem] ;

   for (Index_t i=0; i<numElem; ++i)
      materialID[i] = 1 ;

   DBWrite(fp, "brick_material", materialID, &numElem, 1, DB_INT) ;

   delete [] materialID ;

   DBClose(fp);
}
#endif

#ifdef SAMI
void DumpDomain(Domain *domain)
{
   char meshName[64] ;
   printf("Dumping SAMI file\n");
   sprintf(meshName, "sedov_%d.sami", int(domain->cycle)) ;

   DumpSAMI(domain, meshName) ;
   
}
#endif

void write_solution(Domain* locDom)
{
  Vector_h<Real_t> x_h = locDom->x;
  Vector_h<Real_t> y_h = locDom->y;
  Vector_h<Real_t> z_h = locDom->z;

//  printf("Writing solution to file xyz.asc\n");
  std::stringstream filename;
  filename << "xyz.asc";

  FILE *fout = fopen(filename.str().c_str(),"wb");

  for (Index_t i=0; i<locDom->numNode; i++) {
      fprintf(fout,"%10d\n",i);
      fprintf(fout,"%.10f\n",x_h[i]);
      fprintf(fout,"%.10f\n",y_h[i]);
      fprintf(fout,"%.10f\n",z_h[i]);
  }
  fclose(fout);
}

///////////////////////////////////////////////////////////////////////////
void InitMeshDecomp(Int_t numRanks, Int_t myRank,
                    Int_t *col, Int_t *row, Int_t *plane, Int_t *side)
{
   Int_t testProcs;
   Int_t dx, dy, dz;
   Int_t myDom;
   
   // Assume cube processor layout for now 
   testProcs = Int_t(cbrt(Real_t(numRanks))+0.5) ;
   if (testProcs*testProcs*testProcs != numRanks) {
      printf("Num processors must be a cube of an integer (1, 8, 27, ...)\n") ;
#if USE_MPI      
      MPI_Abort(MPI_COMM_WORLD, -1) ;
#else
      exit(-1);
#endif
   }
   if (sizeof(Real_t) != 4 && sizeof(Real_t) != 8) {
      printf("MPI operations only support float and double right now...\n");
#if USE_MPI      
      MPI_Abort(MPI_COMM_WORLD, -1) ;
#else
      exit(-1);
#endif
   }
   if (MAX_FIELDS_PER_MPI_COMM > CACHE_COHERENCE_PAD_REAL) {
      printf("corner element comm buffers too small.  Fix code.\n") ;
#if USE_MPI      
      MPI_Abort(MPI_COMM_WORLD, -1) ;
#else
      exit(-1);
#endif
   }

   dx = testProcs ;
   dy = testProcs ;
   dz = testProcs ;

   // temporary test
   if (dx*dy*dz != numRanks) {
      printf("error -- must have as many domains as procs\n") ;
#if USE_MPI      
      MPI_Abort(MPI_COMM_WORLD, -1) ;
#else
      exit(-1);
#endif
   }
   Int_t remainder = dx*dy*dz % numRanks ;
   if (myRank < remainder) {
      myDom = myRank*( 1+ (dx*dy*dz / numRanks)) ;
   }
   else {
      myDom = remainder*( 1+ (dx*dy*dz / numRanks)) +
         (myRank - remainder)*(dx*dy*dz/numRanks) ;
   }

   *col = myDom % dx ;
   *row = (myDom / dx) % dy ;
   *plane = myDom / (dx*dy) ;
   *side = testProcs;

   return;
}

void VerifyAndWriteFinalOutput(Real_t elapsed_time,
                               Domain& locDom,
			       Int_t its,
                               Int_t nx,
                               Int_t numRanks, 
                               bool structured)
{
   size_t free_mem, total_mem, used_mem;
   cudaMemGetInfo(&free_mem, &total_mem);
   used_mem= total_mem - free_mem;
#if LULESH_SHOW_PROGRESS == 0
   printf("   Used Memory         =  %8.4f Mb\n", used_mem / (1024.*1024.) );
#endif

   // GrindTime1 only takes a single domain into account, and is thus a good way to measure
   // processor speed indepdendent of MPI parallelism.
   // GrindTime2 takes into account speedups from MPI parallelism 
   Real_t grindTime1; 
   Real_t grindTime2;
   if(structured)
   {
      grindTime1 = ((elapsed_time*1e6)/its)/(nx*nx*nx);
      grindTime2 = ((elapsed_time*1e6)/its)/(nx*nx*nx*numRanks);
   }
   else
   {
      grindTime1 = ((elapsed_time*1e6)/its)/(locDom.numElem);
      grindTime2 = ((elapsed_time*1e6)/its)/(locDom.numElem*numRanks);
   }
   // Copy Energy back to Host 
   if(structured)
   {
      Real_t e_zero;
      Real_t* d_ezero_ptr = locDom.e.raw() + locDom.octantCorner; /* octant corner supposed to be 0 */
      cudaMemcpy(&e_zero, d_ezero_ptr, sizeof(Real_t), cudaMemcpyDeviceToHost);

      printf("Run completed:  \n");
      printf("   Problem size        =  %i \n",    nx);
      printf("   MPI tasks           =  %i \n",    numRanks);
      printf("   Iteration count     =  %i \n",    its);
      printf("   Final Origin Energy = %12.6e \n", e_zero);

      Real_t   MaxAbsDiff = Real_t(0.0);
      Real_t TotalAbsDiff = Real_t(0.0);
      Real_t   MaxRelDiff = Real_t(0.0);

      Real_t *e_all = new Real_t[nx * nx];
      cudaMemcpy(e_all, locDom.e.raw(), nx * nx * sizeof(Real_t), cudaMemcpyDeviceToHost);
      for (Index_t j=0; j<nx; ++j) {
         for (Index_t k=j+1; k<nx; ++k) {
            Real_t AbsDiff = FABS(e_all[j*nx+k]-e_all[k*nx+j]);
            TotalAbsDiff  += AbsDiff;

            if (MaxAbsDiff <AbsDiff) MaxAbsDiff = AbsDiff;

            Real_t RelDiff = AbsDiff / e_all[k*nx+j];

            if (MaxRelDiff <RelDiff)  MaxRelDiff = RelDiff;
         }
      }
      delete e_all;

      // Quick symmetry check
      printf("   Testing Plane 0 of Energy Array on rank 0:\n");
      printf("        MaxAbsDiff   = %12.6e\n",   MaxAbsDiff   );
      printf("        TotalAbsDiff = %12.6e\n",   TotalAbsDiff );
      printf("        MaxRelDiff   = %12.6e\n\n", MaxRelDiff   );
   }

   // Timing information
   printf("\nElapsed time         = %10.2f (s)\n", elapsed_time);
   printf("Grind time (us/z/c)  = %10.8g (per dom)  (%10.8g overall)\n", grindTime1, grindTime2);
   printf("FOM                  = %10.8g (z/s)\n\n", 1000.0/grindTime2); // zones per second

   bool write_solution_flag=true;
   if (write_solution_flag) {
     write_solution(&locDom);
   }

   return ;
}

int main(int argc, char *argv[])
{
  if (argc < 3) {
    printUsage(argv);
    exit( LFileError );
  }
  
  if ( strcmp(argv[1],"-u") != 0 && strcmp(argv[1],"-s") != 0 ) 
  {
    printUsage(argv);
    exit( LFileError ) ;
  }

  int num_iters = -1;
  if (argc == 5) {
    num_iters = atoi(argv[4]);
  }

  bool structured = ( strcmp(argv[1],"-s") == 0 );

  Int_t numRanks ;
  Int_t myRank ;

#if USE_MPI   
  Domain_member fieldData ;

  MPI_Init(&argc, &argv) ;
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks) ;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;
#else
  numRanks = 1;
  myRank = 0;
#endif

  cuda_init(myRank);

  /* assume cube subdomain geometry for now */
  Index_t nx = atoi(argv[2]);

  Domain *locDom ;

  // Set up the mesh and decompose. Assumes regular cubes for now
  Int_t col, row, plane, side;
  InitMeshDecomp(numRanks, myRank, &col, &row, &plane, &side);

  // TODO: change default nr to 11
  Int_t nr = 11;
  Int_t balance = 1;
  Int_t cost = 1;

  // TODO: modify this constructor to account for new fields
  // TODO: setup communication buffers
  locDom = NewDomain(argv, numRanks, col, row, plane, nx, side, structured, nr, balance, cost); 

#if USE_MPI   
   // copy to the host for mpi transfer
   locDom->h_nodalMass = locDom->nodalMass;

   fieldData = &Domain::get_nodalMass;

   // Initial domain boundary communication 
   CommRecv(*locDom, MSG_COMM_SBN, 1,
            locDom->sizeX + 1, locDom->sizeY + 1, locDom->sizeZ + 1,
            true, false) ;
   CommSend(*locDom, MSG_COMM_SBN, 1, &fieldData,
            locDom->sizeX + 1, locDom->sizeY + 1, locDom->sizeZ + 1,
            true, false) ;
   CommSBN(*locDom, 1, &fieldData) ;

   // copy back to the device
   locDom->nodalMass = locDom->h_nodalMass;

   // End initialization
   MPI_Barrier(MPI_COMM_WORLD);
#endif

  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  /* timestep to solution */
  int its=0;

  if (myRank == 0) {
    if (structured)
      printf("Running until t=%f, Problem size=%dx%dx%d\n",locDom->stoptime,nx,nx,nx);
    else 
      printf("Running until t=%f, Problem size=%d \n",locDom->stoptime,locDom->numElem);
  }

  cudaProfilerStart();

#if USE_MPI   
   double start = MPI_Wtime();
#else
   timeval start;
   gettimeofday(&start, NULL) ;
#endif

  while(locDom->time_h < locDom->stoptime)
  {
    // this has been moved after computation of volume forces to hide launch latencies
    //TimeIncrement(locDom) ;

    LagrangeLeapFrog(locDom) ;

    checkErrors(locDom,its,myRank);

    #if LULESH_SHOW_PROGRESS
     if (myRank == 0) 
	 printf("cycle = %d, time = %e, dt=%e\n", its+1, double(locDom->time_h), double(locDom->deltatime_h) ) ;
    #endif
    its++;
    if (its == num_iters) break;
  }

  // make sure GPU finished its work
  cudaDeviceSynchronize();

// Use reduced max elapsed time
   double elapsed_time;
#if USE_MPI   
   elapsed_time = MPI_Wtime() - start;
#else
   timeval end;
   gettimeofday(&end, NULL) ;
   elapsed_time = (double)(end.tv_sec - start.tv_sec) + ((double)(end.tv_usec - start.tv_usec))/1000000 ;
#endif

   double elapsed_timeG;
#if USE_MPI   
   MPI_Reduce(&elapsed_time, &elapsed_timeG, 1, MPI_DOUBLE,
              MPI_MAX, 0, MPI_COMM_WORLD);
#else
   elapsed_timeG = elapsed_time;
#endif

  cudaProfilerStop();

  if (myRank == 0) 
    VerifyAndWriteFinalOutput(elapsed_timeG, *locDom, its, nx, numRanks, structured);

#ifdef SAMI
  DumpDomain(locDom) ;
#endif
  cudaDeviceReset();

#if USE_MPI
   MPI_Finalize() ;
#endif

  return 0 ;
}

