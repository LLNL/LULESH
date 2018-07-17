
// If no MPI, then this whole file is stubbed out

#if USE_MPI
#include <mpi.h>
#include <string.h>
#endif

#include "lulesh.h"

#if USE_MPI
/* Comm Routines */

#define ALLOW_UNPACKED_PLANE false
#define ALLOW_UNPACKED_ROW   false
#define ALLOW_UNPACKED_COL   false

/*
   There are coherence issues for packing and unpacking message
   buffers.  Ideally, you would like a lot of threads to 
   cooperate in the assembly/dissassembly of each message.
   To do that, each thread should really be operating in a
   different coherence zone.

   Let's assume we have three fields, f1 through f3, defined on
   a 61x61x61 cube.  If we want to send the block boundary
   information for each field to each neighbor processor across
   each cube face, then we have three cases for the
   memory layout/coherence of data on each of the six cube
   boundaries:

      (a) Two of the faces will be in contiguous memory blocks
      (b) Two of the faces will be comprised of pencils of
          contiguous memory.
      (c) Two of the faces will have large strides between
          every value living on the face.

   How do you pack and unpack this data in buffers to
   simultaneous achieve the best memory efficiency and
   the most thread independence?

   Do do you pack field f1 through f3 tighly to reduce message
   size?  Do you align each field on a cache coherence boundary
   within the message so that threads can pack and unpack each
   field independently?  For case (b), do you align each
   boundary pencil of each field separately?  This increases
   the message size, but could improve cache coherence so
   each pencil could be processed independently by a separate
   thread with no conflicts.

   Also, memory access for case (c) would best be done without
   going through the cache (the stride is so large it just causes
   a lot of useless cache evictions).  Is it worth creating
   a special case version of the packing algorithm that uses
   non-coherent load/store opcodes?
*/

/******************************************/

template<int type>
__global__ void SendPlane(Real_t *destAddr, Real_t *srcAddr, Index_t sendCount, Index_t dx, Index_t dy, Index_t dz)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= sendCount) return;

  int i, j;

  switch (type) {
  case 0:
    i = tid;
    destAddr[i] = srcAddr[i] ;
    break;
  case 1:
    i = tid;
    destAddr[i] = srcAddr[dx*dy*(dz - 1) + i] ;
    break;
  case 2:
    i = tid / dx;
    j = tid % dx;
    destAddr[i*dx+j] = srcAddr[i*dx*dy + j] ;
    break;
  case 3:
    i = tid / dx;
    j = tid % dx;
    destAddr[i*dx+j] = srcAddr[dx*(dy - 1) + i*dx*dy + j] ;
    break;
  case 4:
    i = tid / dy;
    j = tid % dy;
    destAddr[i*dy + j] = srcAddr[i*dx*dy + j*dx] ;
    break;
  case 5:
    i = tid / dy;
    j = tid % dy;
    destAddr[i*dy + j] = srcAddr[dx - 1 + i*dx*dy + j*dx] ;
    break;
  }
}

template<int type>
__global__ void AddPlane(Real_t *srcAddr, Real_t *destAddr, Index_t recvCount, Index_t dx, Index_t dy, Index_t dz)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= recvCount) return;

  int i, j;

  switch (type) {
  case 0:
    i = tid;
    destAddr[i] += srcAddr[i] ;
    break;
  case 1:
    i = tid;
    destAddr[dx*dy*(dz - 1) + i] += srcAddr[i] ;
    break;
  case 2:
    i = tid / dx;
    j = tid % dx;
    destAddr[i*dx*dy + j] += srcAddr[i*dx + j] ;
    break;
  case 3:
    i = tid / dx;
    j = tid % dx;
    destAddr[dx*(dy - 1) + i*dx*dy + j] += srcAddr[i*dx + j] ;
    break;
  case 4:
    i = tid / dy;
    j = tid % dy;
    destAddr[i*dx*dy + j*dx] += srcAddr[i*dy + j] ;
    break;
  case 5:
    i = tid / dy;
    j = tid % dy;
    destAddr[dx - 1 + i*dx*dy + j*dx] += srcAddr[i*dy + j] ;
    break;
  }
}

template<int type>
__global__ void CopyPlane(Real_t *srcAddr, Real_t *destAddr, Index_t recvCount, Index_t dx, Index_t dy, Index_t dz)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= recvCount) return;

  int i, j;

  switch (type) {
  case 0:
    i = tid;
    destAddr[i] = srcAddr[i] ;
    break;
  case 1:
    i = tid;
    destAddr[dx*dy*(dz - 1) + i] = srcAddr[i] ;
    break;
  case 2:
    i = tid / dx;
    j = tid % dx;
    destAddr[i*dx*dy + j] = srcAddr[i*dx + j] ;
    break;
  case 3:
    i = tid / dx;
    j = tid % dx;
    destAddr[dx*(dy - 1) + i*dx*dy + j] = srcAddr[i*dx + j] ;
    break;
  case 4:
    i = tid / dy;
    j = tid % dy;
    destAddr[i*dx*dy + j*dx] = srcAddr[i*dy + j] ;
    break;
  case 5:
    i = tid / dy;
    j = tid % dy;
    destAddr[dx - 1 + i*dx*dy + j*dx] = srcAddr[i*dy + j] ;
    break;
  }
}

template<int type>
__global__ void SendEdge(Real_t *destAddr, Real_t *srcAddr, Index_t sendCount, Index_t dx, Index_t dy, Index_t dz)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= sendCount) return;

  switch (type) {
  case 0:
    destAddr[i] = srcAddr[i*dx*dy] ;
    break;
  case 1:
    destAddr[i] = srcAddr[i] ;
    break;
  case 2:
    destAddr[i] = srcAddr[i*dx] ;
    break;
  case 3:
    destAddr[i] = srcAddr[dx*dy - 1 + i*dx*dy] ;
    break;
  case 4:
    destAddr[i] = srcAddr[dx*(dy-1) + dx*dy*(dz-1) + i] ;
    break;
  case 5:
    destAddr[i] = srcAddr[dx*dy*(dz-1) + dx - 1 + i*dx] ;
    break;
  case 6:
    destAddr[i] = srcAddr[dx*(dy-1) + i*dx*dy] ;
    break;
  case 7:
    destAddr[i] = srcAddr[dx*dy*(dz-1) + i] ;
    break;
  case 8:
    destAddr[i] = srcAddr[dx*dy*(dz-1) + i*dx] ;
    break;
  case 9:
    destAddr[i] = srcAddr[dx - 1 + i*dx*dy] ;
    break;
  case 10:
    destAddr[i] = srcAddr[dx*(dy - 1) + i] ;
    break;
  case 11:
    destAddr[i] = srcAddr[dx - 1 + i*dx] ;
    break;
  }
}

template<int type>
__global__ void AddEdge(Real_t *srcAddr, Real_t *destAddr, Index_t recvCount, Index_t dx, Index_t dy, Index_t dz)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= recvCount) return;

  switch (type) {
  case 0:
    destAddr[i*dx*dy] += srcAddr[i] ;
    break;
  case 1:
    destAddr[i] += srcAddr[i] ;
    break;
  case 2:
    destAddr[i*dx] += srcAddr[i] ;
    break;
  case 3:
    destAddr[dx*dy - 1 + i*dx*dy] += srcAddr[i] ;
    break;
  case 4:
    destAddr[dx*(dy-1) + dx*dy*(dz-1) + i] += srcAddr[i] ;
    break;
  case 5:
    destAddr[dx*dy*(dz-1) + dx - 1 + i*dx] += srcAddr[i] ;
    break;
  case 6:
    destAddr[dx*(dy-1) + i*dx*dy] += srcAddr[i] ;
    break;
  case 7:
    destAddr[dx*dy*(dz-1) + i] += srcAddr[i] ;
    break;
  case 8:
    destAddr[dx*dy*(dz-1) + i*dx] += srcAddr[i] ;
    break;
  case 9:
    destAddr[dx - 1 + i*dx*dy] += srcAddr[i] ;
    break;
  case 10:
    destAddr[dx*(dy - 1) + i] += srcAddr[i] ;
    break;
  case 11:
    destAddr[dx - 1 + i*dx] += srcAddr[i] ;
    break;
  }
}

template<int type>
__global__ void CopyEdge(Real_t *srcAddr, Real_t *destAddr, Index_t recvCount, Index_t dx, Index_t dy, Index_t dz)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= recvCount) return;

  switch (type) {
  case 0:
    destAddr[i*dx*dy] = srcAddr[i] ;
    break;
  case 1:
    destAddr[i] = srcAddr[i] ;
    break;
  case 2:
    destAddr[i*dx] = srcAddr[i] ;
    break;
  case 3:
    destAddr[dx*dy - 1 + i*dx*dy] = srcAddr[i] ;
    break;
  case 4:
    destAddr[dx*(dy-1) + dx*dy*(dz-1) + i] = srcAddr[i] ;
    break;
  case 5:
    destAddr[dx*dy*(dz-1) + dx - 1 + i*dx] = srcAddr[i] ;
    break;
  case 6:
    destAddr[dx*(dy-1) + i*dx*dy] = srcAddr[i] ;
    break;
  case 7:
    destAddr[dx*dy*(dz-1) + i] = srcAddr[i] ;
    break;
  case 8:
    destAddr[dx*dy*(dz-1) + i*dx] = srcAddr[i] ;
    break;
  case 9:
    destAddr[dx - 1 + i*dx*dy] = srcAddr[i] ;
    break;
  case 10:
    destAddr[dx*(dy - 1) + i] = srcAddr[i] ;
    break;
  case 11:
    destAddr[dx - 1 + i*dx] = srcAddr[i] ;
    break;
  }
}

__global__ void AddCorner(Real_t *destAddr, Real_t src)
{
  destAddr[0] += src;
}

__global__ void CopyCorner(Real_t *destAddr, Real_t src)
{
  destAddr[0] = src;
}

/******************************************/

void CommSendGpu(Domain& domain, int msgType,
              Index_t xferFields, Domain_member *fieldData,
              Index_t dx, Index_t dy, Index_t dz, bool doSend, bool planeOnly, cudaStream_t stream)
{

   if (domain.numRanks() == 1)
      return ;

   /* post recieve buffers for all incoming messages */
   int myRank ;
   Index_t maxPlaneComm = xferFields * domain.maxPlaneSize ;
   Index_t maxEdgeComm  = xferFields * domain.maxEdgeSize ;
   Index_t pmsg = 0 ; /* plane comm msg */
   Index_t emsg = 0 ; /* edge comm msg */
   Index_t cmsg = 0 ; /* corner comm msg */
   MPI_Datatype baseType = ((sizeof(Real_t) == 4) ? MPI_FLOAT : MPI_DOUBLE) ;
   MPI_Status status[26] ;
   Real_t *destAddr ;
   Real_t *d_destAddr ;
   bool rowMin, rowMax, colMin, colMax, planeMin, planeMax ;
   /* assume communication to 6 neighbors by default */
   rowMin = rowMax = colMin = colMax = planeMin = planeMax = true ;
   if (domain.rowLoc() == 0) {
      rowMin = false ;
   }
   if (domain.rowLoc() == (domain.tp()-1)) {
      rowMax = false ;
   }
   if (domain.colLoc() == 0) {
      colMin = false ;
   }
   if (domain.colLoc() == (domain.tp()-1)) {
      colMax = false ;
   }
   if (domain.planeLoc() == 0) {
      planeMin = false ;
   }
   if (domain.planeLoc() == (domain.tp()-1)) {
      planeMax = false ;
   }

   for (Index_t i=0; i<26; ++i) {
      domain.sendRequest[i] = MPI_REQUEST_NULL ;
   }

   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;

   // setup launch grid
   const int block = 128;

   /* post sends */

   if (planeMin | planeMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      int sendCount = dx * dy ;

      if (planeMin) {
	 destAddr = &domain.commDataSend[pmsg * maxPlaneComm] ;
         d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm] ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	    SendPlane<0><<<(sendCount+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), sendCount, dx, dy, dz);
            d_destAddr += sendCount ;
         }
         d_destAddr -= xferFields*sendCount ;
         cudaMemcpyAsync(destAddr, d_destAddr, xferFields*sendCount*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         cudaStreamSynchronize(stream);

         MPI_Isend(destAddr, xferFields*sendCount, baseType,
                   myRank - domain.tp()*domain.tp(), msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg]) ;
         ++pmsg ;
      }
      if (planeMax && doSend) {
         destAddr = &domain.commDataSend[pmsg * maxPlaneComm] ;
	 d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm] ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	    SendPlane<1><<<(sendCount+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), sendCount, dx, dy, dz);
            d_destAddr += sendCount ;
         }
         d_destAddr -= xferFields*sendCount ;
         cudaMemcpyAsync(destAddr, d_destAddr, xferFields*sendCount*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         cudaStreamSynchronize(stream);

         MPI_Isend(destAddr, xferFields*sendCount, baseType,
                   myRank + domain.tp()*domain.tp(), msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg]) ;
         ++pmsg ;
      }
   }
   if (rowMin | rowMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      int sendCount = dx * dz ;

      if (rowMin) {
         destAddr = &domain.commDataSend[pmsg * maxPlaneComm] ;
         d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	    SendPlane<2><<<(sendCount+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), sendCount, dx, dy, dz);
            d_destAddr += sendCount ;
         }
         d_destAddr -= xferFields*sendCount ;
         cudaMemcpyAsync(destAddr, d_destAddr, xferFields*sendCount*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         cudaStreamSynchronize(stream);

         MPI_Isend(destAddr, xferFields*sendCount, baseType,
                   myRank - domain.tp(), msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg]) ;
         ++pmsg ;
      }
      if (rowMax && doSend) {
         destAddr = &domain.commDataSend[pmsg * maxPlaneComm] ;
         d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	    SendPlane<3><<<(sendCount+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), sendCount, dx, dy, dz);
            d_destAddr += sendCount ;
         }
         d_destAddr -= xferFields*sendCount ;
         cudaMemcpyAsync(destAddr, d_destAddr, xferFields*sendCount*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         cudaStreamSynchronize(stream);

         MPI_Isend(destAddr, xferFields*sendCount, baseType,
                   myRank + domain.tp(), msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg]) ;
         ++pmsg ;
      }
   }
   if (colMin | colMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      int sendCount = dy * dz ;

      if (colMin) {
         destAddr = &domain.commDataSend[pmsg * maxPlaneComm] ;
         d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	    SendPlane<4><<<(sendCount+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), sendCount, dx, dy, dz);
            d_destAddr += sendCount ;
         }
         d_destAddr -= xferFields*sendCount ;
         cudaMemcpyAsync(destAddr, d_destAddr, xferFields*sendCount*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         cudaStreamSynchronize(stream);

         MPI_Isend(destAddr, xferFields*sendCount, baseType,
                   myRank - 1, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg]) ;
         ++pmsg ;
      }
      if (colMax && doSend) {
         destAddr = &domain.commDataSend[pmsg * maxPlaneComm] ;
         d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	    SendPlane<5><<<(sendCount+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), sendCount, dx, dy, dz);
            d_destAddr += sendCount ;
         }
         d_destAddr -= xferFields*sendCount ;
         cudaMemcpyAsync(destAddr, d_destAddr, xferFields*sendCount*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         cudaStreamSynchronize(stream);

         MPI_Isend(destAddr, xferFields*sendCount, baseType,
                   myRank + 1, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg]) ;
         ++pmsg ;
      }
   }

   if (!planeOnly) {
      if (rowMin && colMin) {
         int toRank = myRank - domain.tp() - 1 ;
         destAddr = &domain.commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	    SendEdge<0><<<(dz+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), dz, dx, dy, dz);
            d_destAddr += dz ;
         }
         d_destAddr -= xferFields*dz ;
         cudaMemcpyAsync(destAddr, d_destAddr, xferFields*dz*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         cudaStreamSynchronize(stream);

         MPI_Isend(destAddr, xferFields*dz, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      if (rowMin && planeMin) {
         int toRank = myRank - domain.tp()*domain.tp() - domain.tp() ;
         destAddr = &domain.commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	    SendEdge<1><<<(dx+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), dx, dx, dy, dz);
            d_destAddr += dx ;
         }
         d_destAddr -= xferFields*dx ;
         cudaMemcpyAsync(destAddr, d_destAddr, xferFields*dx*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         cudaStreamSynchronize(stream);

         MPI_Isend(destAddr, xferFields*dx, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      if (colMin && planeMin) {
         int toRank = myRank - domain.tp()*domain.tp() - 1 ;
         destAddr = &domain.commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	    SendEdge<2><<<(dy+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), dy, dx, dy, dz);
            d_destAddr += dy ;
         }
         d_destAddr -= xferFields*dy ;
         cudaMemcpyAsync(destAddr, d_destAddr, xferFields*dy*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         cudaStreamSynchronize(stream);

         MPI_Isend(destAddr, xferFields*dy, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      if (rowMax && colMax && doSend) {
         int toRank = myRank + domain.tp() + 1 ;
         destAddr = &domain.commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	    SendEdge<3><<<(dz+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), dz, dx, dy, dz);
            d_destAddr += dz ;
         }
         d_destAddr -= xferFields*dz ;
         cudaMemcpyAsync(destAddr, d_destAddr, xferFields*dz*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         cudaStreamSynchronize(stream);

         MPI_Isend(destAddr, xferFields*dz, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      if (rowMax && planeMax && doSend) {
         int toRank = myRank + domain.tp()*domain.tp() + domain.tp() ;
         destAddr = &domain.commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	    SendEdge<4><<<(dx+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), dx, dx, dy, dz);
            d_destAddr += dx ;
         }
         d_destAddr -= xferFields*dx ;
         cudaMemcpyAsync(destAddr, d_destAddr, xferFields*dx*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         cudaStreamSynchronize(stream);

         MPI_Isend(destAddr, xferFields*dx, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      if (colMax && planeMax && doSend) {
         int toRank = myRank + domain.tp()*domain.tp() + 1 ;
         destAddr = &domain.commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	    SendEdge<5><<<(dy+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), dy, dx, dy, dz);
            d_destAddr += dy ;
         }
         d_destAddr -= xferFields*dy ;
         cudaMemcpyAsync(destAddr, d_destAddr, xferFields*dy*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         cudaStreamSynchronize(stream);

         MPI_Isend(destAddr, xferFields*dy, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      if (rowMax && colMin && doSend) {
         int toRank = myRank + domain.tp() - 1 ;
         destAddr = &domain.commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	    SendEdge<6><<<(dz+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), dz, dx, dy, dz);
            d_destAddr += dz ;
         }
         d_destAddr -= xferFields*dz ;
         cudaMemcpyAsync(destAddr, d_destAddr, xferFields*dz*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         cudaStreamSynchronize(stream);

         MPI_Isend(destAddr, xferFields*dz, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      if (rowMin && planeMax && doSend) {
         int toRank = myRank + domain.tp()*domain.tp() - domain.tp() ;
         destAddr = &domain.commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	    SendEdge<7><<<(dx+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), dx, dx, dy, dz);
            d_destAddr += dx ;
         }
         d_destAddr -= xferFields*dx ;
         cudaMemcpyAsync(destAddr, d_destAddr, xferFields*dx*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         cudaStreamSynchronize(stream);

         MPI_Isend(destAddr, xferFields*dx, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      if (colMin && planeMax && doSend) {
         int toRank = myRank + domain.tp()*domain.tp() - 1 ;
         destAddr = &domain.commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	    SendEdge<8><<<(dy+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), dy, dx, dy, dz);
            d_destAddr += dy ;
         }
         d_destAddr -= xferFields*dy ;
         cudaMemcpyAsync(destAddr, d_destAddr, xferFields*dy*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         cudaStreamSynchronize(stream);

         MPI_Isend(destAddr, xferFields*dy, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      if (rowMin && colMax) {
         int toRank = myRank - domain.tp() + 1 ;
         destAddr = &domain.commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	    SendEdge<9><<<(dz+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), dz, dx, dy, dz);
            d_destAddr += dz ;
         }
         d_destAddr -= xferFields*dz ;
         cudaMemcpyAsync(destAddr, d_destAddr, xferFields*dz*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         cudaStreamSynchronize(stream);

         MPI_Isend(destAddr, xferFields*dz, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      if (rowMax && planeMin) {
         int toRank = myRank - domain.tp()*domain.tp() + domain.tp() ;
         destAddr = &domain.commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	    SendEdge<10><<<(dx+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), dx, dx, dy, dz);
            d_destAddr += dx ;
         }
         d_destAddr -= xferFields*dx ;
         cudaMemcpyAsync(destAddr, d_destAddr, xferFields*dx*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         cudaStreamSynchronize(stream);

         MPI_Isend(destAddr, xferFields*dx, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      if (colMax && planeMin) {
         int toRank = myRank - domain.tp()*domain.tp() + 1 ;
         destAddr = &domain.commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         d_destAddr = &domain.d_commDataSend[pmsg * maxPlaneComm +
                                          emsg * maxEdgeComm] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            Domain_member src = fieldData[fi] ;
	    SendEdge<11><<<(dy+block-1)/block,block,0,stream>>>(d_destAddr, &(domain.*src)(0), dy, dx, dy, dz);
            d_destAddr += dy ;
         }
         d_destAddr -= xferFields*dy ;
         cudaMemcpyAsync(destAddr, d_destAddr, xferFields*dy*sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         cudaStreamSynchronize(stream);

         MPI_Isend(destAddr, xferFields*dy, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
         ++emsg ;
      }

      if (rowMin && colMin && planeMin) {
         /* corner at domain logical coord (0, 0, 0) */
         int toRank = myRank - domain.tp()*domain.tp() - domain.tp() - 1 ;
         Real_t *comBuf = &domain.commDataSend[pmsg * maxPlaneComm +
                                                emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            cudaMemcpyAsync(&comBuf[fi], &(domain.*fieldData[fi])(0), sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         }
         cudaStreamSynchronize(stream);
         MPI_Isend(comBuf, xferFields, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg]) ;
         ++cmsg ;
      }
      if (rowMin && colMin && planeMax && doSend) {
         /* corner at domain logical coord (0, 0, 1) */
         int toRank = myRank + domain.tp()*domain.tp() - domain.tp() - 1 ;
         Real_t *comBuf = &domain.commDataSend[pmsg * maxPlaneComm +
                                                emsg * maxEdgeComm +
                                         cmsg * CACHE_COHERENCE_PAD_REAL] ;
         Index_t idx = dx*dy*(dz - 1) ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            cudaMemcpyAsync(&comBuf[fi], &(domain.*fieldData[fi])(idx), sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         }
         cudaStreamSynchronize(stream);
         MPI_Isend(comBuf, xferFields, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg]) ;
         ++cmsg ;
      }
      if (rowMin && colMax && planeMin) {
         /* corner at domain logical coord (1, 0, 0) */
         int toRank = myRank - domain.tp()*domain.tp() - domain.tp() + 1 ;
         Real_t *comBuf = &domain.commDataSend[pmsg * maxPlaneComm +
                                                emsg * maxEdgeComm +
                                         cmsg * CACHE_COHERENCE_PAD_REAL] ;
         Index_t idx = dx - 1 ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            cudaMemcpyAsync(&comBuf[fi], &(domain.*fieldData[fi])(idx), sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         }
         cudaStreamSynchronize(stream);
         MPI_Isend(comBuf, xferFields, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg]) ;
         ++cmsg ;
      }
      if (rowMin && colMax && planeMax && doSend) {
         /* corner at domain logical coord (1, 0, 1) */
         int toRank = myRank + domain.tp()*domain.tp() - domain.tp() + 1 ;
         Real_t *comBuf = &domain.commDataSend[pmsg * maxPlaneComm +
                                                emsg * maxEdgeComm +
                                         cmsg * CACHE_COHERENCE_PAD_REAL] ;
         Index_t idx = dx*dy*(dz - 1) + (dx - 1) ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            cudaMemcpyAsync(&comBuf[fi], &(domain.*fieldData[fi])(idx), sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         }
         cudaStreamSynchronize(stream);
         MPI_Isend(comBuf, xferFields, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg]) ;
         ++cmsg ;
      }
      if (rowMax && colMin && planeMin) {
         /* corner at domain logical coord (0, 1, 0) */
         int toRank = myRank - domain.tp()*domain.tp() + domain.tp() - 1 ;
         Real_t *comBuf = &domain.commDataSend[pmsg * maxPlaneComm +
                                                emsg * maxEdgeComm +
                                         cmsg * CACHE_COHERENCE_PAD_REAL] ;
         Index_t idx = dx*(dy - 1) ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            cudaMemcpyAsync(&comBuf[fi], &(domain.*fieldData[fi])(idx), sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         }
         cudaStreamSynchronize(stream);
         MPI_Isend(comBuf, xferFields, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg]) ;
         ++cmsg ;
      }
      if (rowMax && colMin && planeMax && doSend) {
         /* corner at domain logical coord (0, 1, 1) */
         int toRank = myRank + domain.tp()*domain.tp() + domain.tp() - 1 ;
         Real_t *comBuf = &domain.commDataSend[pmsg * maxPlaneComm +
                                                emsg * maxEdgeComm +
                                         cmsg * CACHE_COHERENCE_PAD_REAL] ;
         Index_t idx = dx*dy*(dz - 1) + dx*(dy - 1) ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            cudaMemcpyAsync(&comBuf[fi], &(domain.*fieldData[fi])(idx), sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         }
         cudaStreamSynchronize(stream);
         MPI_Isend(comBuf, xferFields, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg]) ;
         ++cmsg ;
      }
      if (rowMax && colMax && planeMin) {
         /* corner at domain logical coord (1, 1, 0) */
         int toRank = myRank - domain.tp()*domain.tp() + domain.tp() + 1 ;
         Real_t *comBuf = &domain.commDataSend[pmsg * maxPlaneComm +
                                                emsg * maxEdgeComm +
                                         cmsg * CACHE_COHERENCE_PAD_REAL] ;
         Index_t idx = dx*dy - 1 ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            cudaMemcpyAsync(&comBuf[fi], &(domain.*fieldData[fi])(idx), sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         }
         cudaStreamSynchronize(stream);
         MPI_Isend(comBuf, xferFields, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg]) ;
         ++cmsg ;
      }
      if (rowMax && colMax && planeMax && doSend) {
         /* corner at domain logical coord (1, 1, 1) */
         int toRank = myRank + domain.tp()*domain.tp() + domain.tp() + 1 ;
         Real_t *comBuf = &domain.commDataSend[pmsg * maxPlaneComm +
                                                emsg * maxEdgeComm +
                                         cmsg * CACHE_COHERENCE_PAD_REAL] ;
         Index_t idx = dx*dy*dz - 1 ;
         for (Index_t fi=0; fi<xferFields; ++fi) {
            cudaMemcpyAsync(&comBuf[fi], &(domain.*fieldData[fi])(idx), sizeof(Real_t), cudaMemcpyDeviceToHost, stream);
         }
         cudaStreamSynchronize(stream);
         MPI_Isend(comBuf, xferFields, baseType, toRank, msgType,
                   MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg]) ;
         ++cmsg ;
      }
   }

   MPI_Waitall(26, domain.sendRequest, status) ;
}

/******************************************/

void CommSBNGpu(Domain& domain, int xferFields, Domain_member *fieldData, cudaStream_t *streams) {

   if (domain.numRanks() == 1)
      return ;

   /* summation order should be from smallest value to largest */
   /* or we could try out kahan summation! */

   int myRank ;
   Index_t maxPlaneComm = xferFields * domain.maxPlaneSize ;
   Index_t maxEdgeComm  = xferFields * domain.maxEdgeSize ;
   Index_t pmsg = 0 ; /* plane comm msg */
   Index_t emsg = 0 ; /* edge comm msg */
   Index_t cmsg = 0 ; /* corner comm msg */
   Index_t dx = domain.sizeX + 1 ;
   Index_t dy = domain.sizeY + 1 ;
   Index_t dz = domain.sizeZ + 1 ;
   MPI_Status status ;
   Real_t *srcAddr ;
   Real_t *d_srcAddr ;
   Index_t rowMin, rowMax, colMin, colMax, planeMin, planeMax ;
   /* assume communication to 6 neighbors by default */
   rowMin = rowMax = colMin = colMax = planeMin = planeMax = 1 ;
   if (domain.rowLoc() == 0) {
      rowMin = 0 ;
   }
   if (domain.rowLoc() == (domain.tp()-1)) {
      rowMax = 0 ;
   }
   if (domain.colLoc() == 0) {
      colMin = 0 ;
   }
   if (domain.colLoc() == (domain.tp()-1)) {
      colMax = 0 ;
   }
   if (domain.planeLoc() == 0) {
      planeMin = 0 ;
   }
   if (domain.planeLoc() == (domain.tp()-1)) {
      planeMax = 0 ;
   }

   // setup launch grid
   const int block = 128;

   // streams
   int s = 0;
   cudaStream_t stream;

   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;

   if (planeMin | planeMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dx * dy ;

      if (planeMin) {
         /* contiguous memory */
	 stream = streams[s++];
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
	    AddPlane<0><<<(opCount+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), opCount, dx, dy, dz);
            d_srcAddr += opCount ;
         }
         ++pmsg ;
      }
      if (planeMax) {
         /* contiguous memory */
	 stream = streams[s++];
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
	    AddPlane<1><<<(opCount+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), opCount, dx, dy, dz);
            d_srcAddr += opCount ;
         }
         ++pmsg ;
      }
   }

   if (rowMin | rowMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dx * dz ;

      if (rowMin) {
         /* contiguous memory */
	 stream = streams[s++];
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
	    AddPlane<2><<<(opCount+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), opCount, dx, dy, dz);
            d_srcAddr += opCount ;
         }
         ++pmsg ;
      }
      if (rowMax) {
         /* contiguous memory */
	 stream = streams[s++];
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
	    AddPlane<3><<<(opCount+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), opCount, dx, dy, dz);
            d_srcAddr += opCount ;
         }
         ++pmsg ;
      }
   }
   if (colMin | colMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dy * dz ;

      if (colMin) {
         /* contiguous memory */
	 stream = streams[s++];
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
	    AddPlane<4><<<(opCount+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), opCount, dx, dy, dz);
            d_srcAddr += opCount ;
         }
         ++pmsg ;
      }
      if (colMax) {
         /* contiguous memory */
	 stream = streams[s++];
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
	    AddPlane<5><<<(opCount+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), opCount, dx, dy, dz);
            d_srcAddr += opCount ;
         }
         ++pmsg ;
      }
   }

   if (rowMin & colMin) {
      stream = streams[s++];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dz*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 AddEdge<0><<<(dz+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dz, dx, dy, dz);
         d_srcAddr += dz ;
      }
      ++emsg ;
   }

   if (rowMin & planeMin) {
      stream = streams[s++];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dx*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 AddEdge<1><<<(dx+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dx, dx, dy, dz);
         d_srcAddr += dx ;
      }
      ++emsg ;
   }

   if (colMin & planeMin) {
      stream = streams[s++];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dy*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 AddEdge<2><<<(dy+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dy, dx, dy, dz);
         d_srcAddr += dy ;
      }
      ++emsg ;
   }

   if (rowMax & colMax) {
      stream = streams[s++];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dz*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 AddEdge<3><<<(dz+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dz, dx, dy, dz);
         d_srcAddr += dz ;
      }
      ++emsg ;
   }

   if (rowMax & planeMax) {
      stream = streams[s++];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dx*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 AddEdge<4><<<(dx+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dx, dx, dy, dz);
         d_srcAddr += dx ;
      }
      ++emsg ;
   }

   if (colMax & planeMax) {
      stream = streams[s++];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dy*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 AddEdge<5><<<(dy+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dy, dx, dy, dz);
         d_srcAddr += dy ;
      }
      ++emsg ;
   }

   if (rowMax & colMin) {
      stream = streams[s++];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dz*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 AddEdge<6><<<(dz+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dz, dx, dy, dz);
         d_srcAddr += dz ;
      }
      ++emsg ;
   }

   if (rowMin & planeMax) {
      stream = streams[s++];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dx*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 AddEdge<7><<<(dx+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dx, dx, dy, dz);
         d_srcAddr += dx ;
      }
      ++emsg ;
   }

   if (colMin & planeMax) {
      stream = streams[s++];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dy*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 AddEdge<8><<<(dy+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dy, dx, dy, dz);
         d_srcAddr += dy ;
      }
      ++emsg ;
   }

   if (rowMin & colMax) {
      stream = streams[s++];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dz*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 AddEdge<9><<<(dz+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dz, dx, dy, dz);
         d_srcAddr += dz ;
      }
      ++emsg ;
   }

   if (rowMax & planeMin) {
      stream = streams[s++];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dx*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 AddEdge<10><<<(dx+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dx, dx, dy, dz);
         d_srcAddr += dx ;
      }
      ++emsg ;
   }

   if (colMax & planeMin) {
      stream = streams[s++];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dy*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 AddEdge<11><<<(dy+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dy, dx, dy, dz);
         d_srcAddr += dy ;
      }
      ++emsg ;
   }

   if (rowMin & colMin & planeMin) {
      stream = streams[s++];
      /* corner at domain logical coord (0, 0, 0) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         AddCorner<<<1,1,0,stream>>>(&(domain.*fieldData[fi])(0), comBuf[fi]) ;
      }
      ++cmsg ;
   }
   if (rowMin & colMin & planeMax) {
      stream = streams[s++];
      /* corner at domain logical coord (0, 0, 1) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx*dy*(dz - 1) ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         AddCorner<<<1,1,0,stream>>>(&(domain.*fieldData[fi])(idx), comBuf[fi]) ;
      }
      ++cmsg ;
   }
   if (rowMin & colMax & planeMin) {
      stream = streams[s++];
      /* corner at domain logical coord (1, 0, 0) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx - 1 ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         AddCorner<<<1,1,0,stream>>>(&(domain.*fieldData[fi])(idx), comBuf[fi]) ;
      }
      ++cmsg ;
   }
   if (rowMin & colMax & planeMax) {
      stream = streams[s++];
      /* corner at domain logical coord (1, 0, 1) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx*dy*(dz - 1) + (dx - 1) ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         AddCorner<<<1,1,0,stream>>>(&(domain.*fieldData[fi])(idx), comBuf[fi]) ;
      }
      ++cmsg ;
   }
   if (rowMax & colMin & planeMin) {
      stream = streams[s++];
      /* corner at domain logical coord (0, 1, 0) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx*(dy - 1) ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         AddCorner<<<1,1,0,stream>>>(&(domain.*fieldData[fi])(idx), comBuf[fi]) ;
      }
      ++cmsg ;
   }
   if (rowMax & colMin & planeMax) {
      stream = streams[s++];
      /* corner at domain logical coord (0, 1, 1) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx*dy*(dz - 1) + dx*(dy - 1) ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         AddCorner<<<1,1,0,stream>>>(&(domain.*fieldData[fi])(idx), comBuf[fi]) ;
      }
      ++cmsg ;
   }
   if (rowMax & colMax & planeMin) {
      stream = streams[s++];
      /* corner at domain logical coord (1, 1, 0) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx*dy - 1 ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         AddCorner<<<1,1,0,stream>>>(&(domain.*fieldData[fi])(idx), comBuf[fi]) ;
      }
      ++cmsg ;
   }
   if (rowMax & colMax & planeMax) {
      stream = streams[s++];
      /* corner at domain logical coord (1, 1, 1) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx*dy*dz - 1 ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         AddCorner<<<1,1,0,stream>>>(&(domain.*fieldData[fi])(idx), comBuf[fi]) ;
      }
      ++cmsg ;
   }

   // don't need to call synchronize since it will be done automatically 
   // before kernels start to execute in NULL stream
}

/******************************************/

void CommSyncPosVelGpu(Domain& domain, cudaStream_t *streams) {

   if (domain.numRanks() == 1)
      return ;

   int myRank ;
   bool doRecv = false ;
   Index_t xferFields = 6 ; /* x, y, z, xd, yd, zd */
   Domain_member fieldData[6] ;
   Index_t maxPlaneComm = xferFields * domain.maxPlaneSize ;
   Index_t maxEdgeComm  = xferFields * domain.maxEdgeSize ;
   Index_t pmsg = 0 ; /* plane comm msg */
   Index_t emsg = 0 ; /* edge comm msg */
   Index_t cmsg = 0 ; /* corner comm msg */
   Index_t dx = domain.sizeX + 1 ;
   Index_t dy = domain.sizeY + 1 ;
   Index_t dz = domain.sizeZ + 1 ;
   MPI_Status status ;
   Real_t *srcAddr ;
   Real_t *d_srcAddr ;
   bool rowMin, rowMax, colMin, colMax, planeMin, planeMax ;

   /* assume communication to 6 neighbors by default */
   rowMin = rowMax = colMin = colMax = planeMin = planeMax = true ;
   if (domain.rowLoc() == 0) {
      rowMin = false ;
   }
   if (domain.rowLoc() == (domain.tp()-1)) {
      rowMax = false ;
   }
   if (domain.colLoc() == 0) {
      colMin = false ;
   }
   if (domain.colLoc() == (domain.tp()-1)) {
      colMax = false ;
   }
   if (domain.planeLoc() == 0) {
      planeMin = false ;
   }
   if (domain.planeLoc() == (domain.tp()-1)) {
      planeMax = false ;
   }

   fieldData[0] = &Domain::get_x ;
   fieldData[1] = &Domain::get_y ;
   fieldData[2] = &Domain::get_z ;
   fieldData[3] = &Domain::get_xd ;
   fieldData[4] = &Domain::get_yd ;
   fieldData[5] = &Domain::get_zd ;

   // setup launch grid
   const int block = 128;

   // streams
   int s = 0;
   cudaStream_t stream;

   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;

   if (planeMin | planeMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dx * dy ;

      if (planeMin && doRecv) {
         /* contiguous memory */
	 stream = streams[s++];
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
	    CopyPlane<0><<<(opCount+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), opCount, dx, dy, dz);
            d_srcAddr += opCount ;
         }
         ++pmsg ;
      }
      if (planeMax) {
         /* contiguous memory */
	 stream = streams[s++];
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
	    CopyPlane<1><<<(opCount+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), opCount, dx, dy, dz);
            d_srcAddr += opCount ;
         }
         ++pmsg ;
      }
   }

   if (rowMin | rowMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dx * dz ;

      if (rowMin && doRecv) {
         /* contiguous memory */
	 stream = streams[s++];
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
	    CopyPlane<2><<<(opCount+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), opCount, dx, dy, dz);
            d_srcAddr += opCount ;
         }
         ++pmsg ;
      }
      if (rowMax) {
         /* contiguous memory */
	 stream = streams[s++];
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
	    CopyPlane<3><<<(opCount+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), opCount, dx, dy, dz);
            d_srcAddr += opCount ;
         }
         ++pmsg ;
      }
   }
   if (colMin | colMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dy * dz ;

      if (colMin && doRecv) {
         /* contiguous memory */
	 stream = streams[s++];
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
	    CopyPlane<4><<<(opCount+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), opCount, dx, dy, dz);
            d_srcAddr += opCount ;
         }
         ++pmsg ;
      }
      if (colMax) {
         /* contiguous memory */
	 stream = streams[s++];
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
	    CopyPlane<5><<<(opCount+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), opCount, dx, dy, dz);
            d_srcAddr += opCount ;
         }
         ++pmsg ;
      }
   }

   if (rowMin && colMin && doRecv) {
      stream = streams[s++];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dz*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 CopyEdge<0><<<(dz+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dz, dx, dy, dz);
         d_srcAddr += dz ;
      }
      ++emsg ;
   }

   if (rowMin && planeMin && doRecv) {
      stream = streams[s++];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dx*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 CopyEdge<1><<<(dx+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dx, dx, dy, dz);
         d_srcAddr += dx ;
      }
      ++emsg ;
   }

   if (colMin && planeMin && doRecv) {
      stream = streams[s++];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dy*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 CopyEdge<2><<<(dy+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dy, dx, dy, dz);
         d_srcAddr += dy ;
      }
      ++emsg ;
   }

   if (rowMax & colMax) {
      stream = streams[s++];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dz*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 CopyEdge<3><<<(dz+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dz, dx, dy, dz);
         d_srcAddr += dz ;
      }
      ++emsg ;
   }

   if (rowMax & planeMax) {
      stream = streams[s++];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dx*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 CopyEdge<4><<<(dx+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dx, dx, dy, dz);
         d_srcAddr += dx ;
      }
      ++emsg ;
   }

   if (colMax & planeMax) {
      stream = streams[s++];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dy*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 CopyEdge<5><<<(dy+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dy, dx, dy, dz);
         d_srcAddr += dy ;
      }
      ++emsg ;
   }

   if (rowMax & colMin) {
      stream = streams[s++];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dz*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 CopyEdge<6><<<(dz+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dz, dx, dy, dz);
         d_srcAddr += dz ;
      }
      ++emsg ;
   }

   if (rowMin & planeMax) {
      stream = streams[s++];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dx*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 CopyEdge<7><<<(dx+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dx, dx, dy, dz);
         d_srcAddr += dx ;
      }
      ++emsg ;
   }

   if (colMin & planeMax) {
      stream = streams[s++];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dy*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 CopyEdge<8><<<(dy+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dy, dx, dy, dz);
         d_srcAddr += dy ;
      }
      ++emsg ;
   }

   if (rowMin && colMax && doRecv) {
      stream = streams[s++];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dz*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 CopyEdge<9><<<(dz+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dz, dx, dy, dz);
         d_srcAddr += dz ;
      }
      ++emsg ;
   }

   if (rowMax && planeMin && doRecv) {
      stream = streams[s++];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dx*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 CopyEdge<10><<<(dx+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dx, dx, dy, dz);
         d_srcAddr += dx ;
      }
      ++emsg ;
   }

   if (colMax && planeMin && doRecv) {
      stream = streams[s++];
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      d_srcAddr = &domain.d_commDataRecv[pmsg * maxPlaneComm +
                                       emsg * maxEdgeComm] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
      cudaMemcpyAsync(d_srcAddr, srcAddr, xferFields*dy*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Domain_member dest = fieldData[fi] ;
	 CopyEdge<11><<<(dy+block-1)/block,block,0,stream>>>(d_srcAddr, &(domain.*dest)(0), dy, dx, dy, dz);
         d_srcAddr += dy ;
      }
      ++emsg ;
   }

   if (rowMin & colMin & planeMin & doRecv) {
      stream = streams[s++];
      /* corner at domain logical coord (0, 0, 0) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         CopyCorner<<<1,1,0,stream>>>(&(domain.*fieldData[fi])(0), comBuf[fi]) ;
      }
      ++cmsg ;
   }
   if (rowMin & colMin & planeMax) {
      stream = streams[s++];
      /* corner at domain logical coord (0, 0, 1) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx*dy*(dz - 1) ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         CopyCorner<<<1,1,0,stream>>>(&(domain.*fieldData[fi])(idx), comBuf[fi]) ;
      }
      ++cmsg ;
   }
   if (rowMin & colMax & planeMin & doRecv) {
      stream = streams[s++];
      /* corner at domain logical coord (1, 0, 0) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx - 1 ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         CopyCorner<<<1,1,0,stream>>>(&(domain.*fieldData[fi])(idx), comBuf[fi]) ;
      }
      ++cmsg ;
   }
   if (rowMin & colMax & planeMax) {
      stream = streams[s++];
      /* corner at domain logical coord (1, 0, 1) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx*dy*(dz - 1) + (dx - 1) ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         CopyCorner<<<1,1,0,stream>>>(&(domain.*fieldData[fi])(idx), comBuf[fi]) ;
      }
      ++cmsg ;
   }
   if (rowMax & colMin & planeMin & doRecv) {
      stream = streams[s++];
      /* corner at domain logical coord (0, 1, 0) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx*(dy - 1) ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         CopyCorner<<<1,1,0,stream>>>(&(domain.*fieldData[fi])(idx), comBuf[fi]) ;
      }
      ++cmsg ;
   }
   if (rowMax & colMin & planeMax) {
      stream = streams[s++];
      /* corner at domain logical coord (0, 1, 1) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx*dy*(dz - 1) + dx*(dy - 1) ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         CopyCorner<<<1,1,0,stream>>>(&(domain.*fieldData[fi])(idx), comBuf[fi]) ;
      }
      ++cmsg ;
   }
   if (rowMax & colMax & planeMin & doRecv) {
      stream = streams[s++];
      /* corner at domain logical coord (1, 1, 0) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx*dy - 1 ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         CopyCorner<<<1,1,0,stream>>>(&(domain.*fieldData[fi])(idx), comBuf[fi]) ;
      }
      ++cmsg ;
   }
   if (rowMax & colMax & planeMax) {
      stream = streams[s++];
      /* corner at domain logical coord (1, 1, 1) */
      Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
                                             emsg * maxEdgeComm +
                                      cmsg * CACHE_COHERENCE_PAD_REAL] ;
      Index_t idx = dx*dy*dz - 1 ;
      MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
         CopyCorner<<<1,1,0,stream>>>(&(domain.*fieldData[fi])(idx), comBuf[fi]) ;
      }
      ++cmsg ;
   }

   // don't need to call synchronize since it will be done automatically 
   // before kernels start to execute in NULL stream
}

/******************************************/

void CommMonoQGpu(Domain& domain, cudaStream_t stream)
{
   if (domain.numRanks() == 1)
      return ;

   int myRank ;
   Index_t xferFields = 3 ; /* delv_xi, delv_eta, delv_zeta */
   Domain_member fieldData[3] ;
   Index_t fieldOffset[3] ;
   Index_t maxPlaneComm = xferFields * domain.maxPlaneSize ;
   Index_t pmsg = 0 ; /* plane comm msg */
   Index_t dx = domain.sizeX ;
   Index_t dy = domain.sizeY ;
   Index_t dz = domain.sizeZ ;
   MPI_Status status ;
   Real_t *srcAddr ;
   bool rowMin, rowMax, colMin, colMax, planeMin, planeMax ;
   /* assume communication to 6 neighbors by default */
   rowMin = rowMax = colMin = colMax = planeMin = planeMax = true ;
   if (domain.rowLoc() == 0) {
      rowMin = false ;
   }
   if (domain.rowLoc() == (domain.tp()-1)) {
      rowMax = false ;
   }
   if (domain.colLoc() == 0) {
      colMin = false ;
   }
   if (domain.colLoc() == (domain.tp()-1)) {
      colMax = false ;
   }
   if (domain.planeLoc() == 0) {
      planeMin = false ;
   }
   if (domain.planeLoc() == (domain.tp()-1)) {
      planeMax = false ;
   }

   /* point into ghost data area */
   // fieldData[0] = &(domain.delv_xi(domain.numElem())) ;
   // fieldData[1] = &(domain.delv_eta(domain.numElem())) ;
   // fieldData[2] = &(domain.delv_zeta(domain.numElem())) ;
   fieldData[0] = &Domain::get_delv_xi ;
   fieldData[1] = &Domain::get_delv_eta ;
   fieldData[2] = &Domain::get_delv_zeta ;
   fieldOffset[0] = domain.numElem ;
   fieldOffset[1] = domain.numElem ;
   fieldOffset[2] = domain.numElem ;

   MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;

   if (planeMin | planeMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dx * dy ;

      if (planeMin) {
         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            cudaMemcpyAsync(&(domain.*dest)(fieldOffset[fi]), srcAddr, opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
            srcAddr += opCount ;
            fieldOffset[fi] += opCount ;
         }
         ++pmsg ;
      }
      if (planeMax) {
         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            cudaMemcpyAsync(&(domain.*dest)(fieldOffset[fi]), srcAddr, opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
            srcAddr += opCount ;
            fieldOffset[fi] += opCount ;
         }
         ++pmsg ;
      }
   }

   if (rowMin | rowMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dx * dz ;

      if (rowMin) {
         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            cudaMemcpyAsync(&(domain.*dest)(fieldOffset[fi]), srcAddr, opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
            srcAddr += opCount ;
            fieldOffset[fi] += opCount ;
         }
         ++pmsg ;
      }
      if (rowMax) {
         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            cudaMemcpyAsync(&(domain.*dest)(fieldOffset[fi]), srcAddr, opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
            srcAddr += opCount ;
            fieldOffset[fi] += opCount ;
         }
         ++pmsg ;
      }
   }
   if (colMin | colMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      Index_t opCount = dy * dz ;

      if (colMin) {
         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            cudaMemcpyAsync(&(domain.*dest)(fieldOffset[fi]), srcAddr, opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
            srcAddr += opCount ;
            fieldOffset[fi] += opCount ;
         }
         ++pmsg ;
      }
      if (colMax) {
         /* contiguous memory */
         srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
         MPI_Wait(&domain.recvRequest[pmsg], &status) ;
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Domain_member dest = fieldData[fi] ;
            cudaMemcpyAsync(&(domain.*dest)(fieldOffset[fi]), srcAddr, opCount*sizeof(Real_t), cudaMemcpyHostToDevice, stream);
            srcAddr += opCount ;
         }
         ++pmsg ;
      }
   }

   // don't need to call synchronize since it will be done automatically 
   // before kernels start to execute in NULL stream
}

#endif
