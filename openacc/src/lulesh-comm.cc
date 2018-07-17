#include "lulesh.h"

// If no MPI, then this whole file is stubbed out
#if USE_MPI

#include <mpi.h>
#include <string.h>

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

/*
  Currently, all message traffic occurs at once.
  We could spread message traffic out like this: 

  CommRecv(domain) ;
  forall(domain.views()-attr("chunk & boundary")) {
     ... do work in parallel ...
  }
  CommSend(domain) ;
  forall(domain.views()-attr("chunk & ~boundary")) {
     ... do work in parallel ...
  }
  CommSBN() ;

  or the CommSend() could function as a semaphore
  for even finer granularity.  When the last chunk
  on a boundary marks the boundary as complete, the
  send could happen immediately:

  CommRecv(domain) ;
  forall(domain.views()-attr("chunk & boundary")) {
     ... do work in parallel ...
     CommSend(domain) ;
  }
  forall(domain.views()-attr("chunk & ~boundary")) {
     ... do work in parallel ...
  }
  CommSBN() ;
*/

/******************************************/


/* doRecv flag only works with regular block structure */
void CommRecv(Domain& domain, int msgType, Index_t xferFields,
              Index_t dx, Index_t dy, Index_t dz, bool doRecv, bool planeOnly) {

  if (domain.numRanks() == 1)
    return ;

  /* post recieve buffers for all incoming messages */
  int myRank ;
  Index_t maxPlaneComm = xferFields * domain.maxPlaneSize() ;
  Index_t maxEdgeComm  = xferFields * domain.maxEdgeSize() ;
  Index_t pmsg = 0 ; /* plane comm msg */
  Index_t emsg = 0 ; /* edge comm msg */
  Index_t cmsg = 0 ; /* corner comm msg */
  MPI_Datatype baseType = ((sizeof(Real_t) == 4) ? MPI_FLOAT : MPI_DOUBLE) ;
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
    domain.recvRequest[i] = MPI_REQUEST_NULL ;
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;

  /* post receives */

  /* receive data from neighboring domain faces */
  if (planeMin && doRecv) {
    /* contiguous memory */
    int fromRank = myRank - domain.tp()*domain.tp() ;
    int recvCount = dx * dy * xferFields ;
    MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm],
        recvCount, baseType, fromRank, msgType,
        MPI_COMM_WORLD, &domain.recvRequest[pmsg]) ;
    ++pmsg ;
  }
  if (planeMax) {
    /* contiguous memory */
    int fromRank = myRank + domain.tp()*domain.tp() ;
    int recvCount = dx * dy * xferFields ;
    MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm],
        recvCount, baseType, fromRank, msgType,
        MPI_COMM_WORLD, &domain.recvRequest[pmsg]) ;
    ++pmsg ;
  }
  if (rowMin && doRecv) {
    /* semi-contiguous memory */
    int fromRank = myRank - domain.tp() ;
    int recvCount = dx * dz * xferFields ;
    MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm],
        recvCount, baseType, fromRank, msgType,
        MPI_COMM_WORLD, &domain.recvRequest[pmsg]) ;
    ++pmsg ;
  }
  if (rowMax) {
    /* semi-contiguous memory */
    int fromRank = myRank + domain.tp() ;
    int recvCount = dx * dz * xferFields ;
    MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm],
        recvCount, baseType, fromRank, msgType,
        MPI_COMM_WORLD, &domain.recvRequest[pmsg]) ;
    ++pmsg ;
  }
  if (colMin && doRecv) {
    /* scattered memory */
    int fromRank = myRank - 1 ;
    int recvCount = dy * dz * xferFields ;
    MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm],
        recvCount, baseType, fromRank, msgType,
        MPI_COMM_WORLD, &domain.recvRequest[pmsg]) ;
    ++pmsg ;
  }
  if (colMax) {
    /* scattered memory */
    int fromRank = myRank + 1 ;
    int recvCount = dy * dz * xferFields ;
    MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm],
        recvCount, baseType, fromRank, msgType,
        MPI_COMM_WORLD, &domain.recvRequest[pmsg]) ;
    ++pmsg ;
  }

  if (!planeOnly) {
    /* receive data from domains connected only by an edge */
    if (rowMin && colMin && doRecv) {
      int fromRank = myRank - domain.tp() - 1 ;
      MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
          emsg * maxEdgeComm],
          dz * xferFields, baseType, fromRank, msgType,
          MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
      ++emsg ;
    }

    if (rowMin && planeMin && doRecv) {
      int fromRank = myRank - domain.tp()*domain.tp() - domain.tp() ;
      MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
          emsg * maxEdgeComm],
          dx * xferFields, baseType, fromRank, msgType,
          MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
      ++emsg ;
    }

    if (colMin && planeMin && doRecv) {
      int fromRank = myRank - domain.tp()*domain.tp() - 1 ;
      MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
          emsg * maxEdgeComm],
          dy * xferFields, baseType, fromRank, msgType,
          MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
      ++emsg ;
    }

    if (rowMax && colMax) {
      int fromRank = myRank + domain.tp() + 1 ;
      MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
          emsg * maxEdgeComm],
          dz * xferFields, baseType, fromRank, msgType,
          MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
      ++emsg ;
    }

    if (rowMax && planeMax) {
      int fromRank = myRank + domain.tp()*domain.tp() + domain.tp() ;
      MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
          emsg * maxEdgeComm],
          dx * xferFields, baseType, fromRank, msgType,
          MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
      ++emsg ;
    }

    if (colMax && planeMax) {
      int fromRank = myRank + domain.tp()*domain.tp() + 1 ;
      MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
          emsg * maxEdgeComm],
          dy * xferFields, baseType, fromRank, msgType,
          MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
      ++emsg ;
    }

    if (rowMax && colMin) {
      int fromRank = myRank + domain.tp() - 1 ;
      MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
          emsg * maxEdgeComm],
          dz * xferFields, baseType, fromRank, msgType,
          MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
      ++emsg ;
    }

    if (rowMin && planeMax) {
      int fromRank = myRank + domain.tp()*domain.tp() - domain.tp() ;
      MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
          emsg * maxEdgeComm],
          dx * xferFields, baseType, fromRank, msgType,
          MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
      ++emsg ;
    }

    if (colMin && planeMax) {
      int fromRank = myRank + domain.tp()*domain.tp() - 1 ;
      MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
          emsg * maxEdgeComm],
          dy * xferFields, baseType, fromRank, msgType,
          MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
      ++emsg ;
    }

    if (rowMin && colMax && doRecv) {
      int fromRank = myRank - domain.tp() + 1 ;
      MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
          emsg * maxEdgeComm],
          dz * xferFields, baseType, fromRank, msgType,
          MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
      ++emsg ;
    }

    if (rowMax && planeMin && doRecv) {
      int fromRank = myRank - domain.tp()*domain.tp() + domain.tp() ;
      MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
          emsg * maxEdgeComm],
          dx * xferFields, baseType, fromRank, msgType,
          MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
      ++emsg ;
    }

    if (colMax && planeMin && doRecv) {
      int fromRank = myRank - domain.tp()*domain.tp() + 1 ;
      MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
          emsg * maxEdgeComm],
          dy * xferFields, baseType, fromRank, msgType,
          MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg]) ;
      ++emsg ;
    }

    /* receive data from domains connected only by a corner */
    if (rowMin && colMin && planeMin && doRecv) {
      /* corner at domain logical coord (0, 0, 0) */
      int fromRank = myRank - domain.tp()*domain.tp() - domain.tp() - 1 ;
      MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
          emsg * maxEdgeComm +
          cmsg * CACHE_COHERENCE_PAD_REAL],
          xferFields, baseType, fromRank, msgType,
          MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg]) ;
      ++cmsg ;
    }
    if (rowMin && colMin && planeMax) {
      /* corner at domain logical coord (0, 0, 1) */
      int fromRank = myRank + domain.tp()*domain.tp() - domain.tp() - 1 ;
      MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
          emsg * maxEdgeComm +
          cmsg * CACHE_COHERENCE_PAD_REAL],
          xferFields, baseType, fromRank, msgType,
          MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg]) ;
      ++cmsg ;
    }
    if (rowMin && colMax && planeMin && doRecv) {
      /* corner at domain logical coord (1, 0, 0) */
      int fromRank = myRank - domain.tp()*domain.tp() - domain.tp() + 1 ;
      MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
          emsg * maxEdgeComm +
          cmsg * CACHE_COHERENCE_PAD_REAL],
          xferFields, baseType, fromRank, msgType,
          MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg]) ;
      ++cmsg ;
    }
    if (rowMin && colMax && planeMax) {
      /* corner at domain logical coord (1, 0, 1) */
      int fromRank = myRank + domain.tp()*domain.tp() - domain.tp() + 1 ;
      MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
          emsg * maxEdgeComm +
          cmsg * CACHE_COHERENCE_PAD_REAL],
          xferFields, baseType, fromRank, msgType,
          MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg]) ;
      ++cmsg ;
    }
    if (rowMax && colMin && planeMin && doRecv) {
      /* corner at domain logical coord (0, 1, 0) */
      int fromRank = myRank - domain.tp()*domain.tp() + domain.tp() - 1 ;
      MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
          emsg * maxEdgeComm +
          cmsg * CACHE_COHERENCE_PAD_REAL],
          xferFields, baseType, fromRank, msgType,
          MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg]) ;
      ++cmsg ;
    }
    if (rowMax && colMin && planeMax) {
      /* corner at domain logical coord (0, 1, 1) */
      int fromRank = myRank + domain.tp()*domain.tp() + domain.tp() - 1 ;
      MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
          emsg * maxEdgeComm +
          cmsg * CACHE_COHERENCE_PAD_REAL],
          xferFields, baseType, fromRank, msgType,
          MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg]) ;
      ++cmsg ;
    }
    if (rowMax && colMax && planeMin && doRecv) {
      /* corner at domain logical coord (1, 1, 0) */
      int fromRank = myRank - domain.tp()*domain.tp() + domain.tp() + 1 ;
      MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
          emsg * maxEdgeComm +
          cmsg * CACHE_COHERENCE_PAD_REAL],
          xferFields, baseType, fromRank, msgType,
          MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg]) ;
      ++cmsg ;
    }
    if (rowMax && colMax && planeMax) {
      /* corner at domain logical coord (1, 1, 1) */
      int fromRank = myRank + domain.tp()*domain.tp() + domain.tp() + 1 ;
      MPI_Irecv(&domain.commDataRecv[pmsg * maxPlaneComm +
          emsg * maxEdgeComm +
          cmsg * CACHE_COHERENCE_PAD_REAL],
          xferFields, baseType, fromRank, msgType,
          MPI_COMM_WORLD, &domain.recvRequest[pmsg+emsg+cmsg]) ;
      ++cmsg ;
    }
  }
}

/******************************************/

void CommSend(Domain& domain, int msgType,
              Index_t xferFields, Real_t **fieldData,
              Index_t dx, Index_t dy, Index_t dz, bool doSend, bool planeOnly)
{
  if (domain.numRanks() == 1)
    return ;

  /* post recieve buffers for all incoming messages */
  int myRank ;
  Index_t maxPlaneComm = xferFields * domain.maxPlaneSize() ;
  Index_t maxEdgeComm  = xferFields * domain.maxEdgeSize() ;
  Index_t pmsg = 0 ; /* plane comm msg */
  Index_t emsg = 0 ; /* edge comm msg */
  Index_t cmsg = 0 ; /* corner comm msg */
  MPI_Datatype baseType = ((sizeof(Real_t) == 4) ? MPI_FLOAT : MPI_DOUBLE) ;
  MPI_Status status[26] ;
  Real_t *destAddr ;
  bool rowMin, rowMax, colMin, colMax, planeMin, planeMax ;
  bool packable ;
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

  packable = true ;
  for (Index_t i=0; i<xferFields-2; ++i) {
    if((fieldData[i+1] - fieldData[i]) != (fieldData[i+2] - fieldData[i+1])) {
      packable = false ;
      break ;
    }
  }
  for (Index_t i=0; i<26; ++i) {
    domain.sendRequest[i] = MPI_REQUEST_NULL ;
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;

  /* post sends */

  if (planeMin | planeMax) {
    /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
    static MPI_Datatype msgTypePlane ;
    static bool packPlane ;
    int sendCount = dx * dy ;

    if (msgTypePlane == 0) {
      /* Create an MPI_struct for field data */
      if (ALLOW_UNPACKED_PLANE && packable) {

        MPI_Type_vector(xferFields, sendCount,
            (fieldData[1] - fieldData[0]),
            baseType, &msgTypePlane) ;
        MPI_Type_commit(&msgTypePlane) ;
        packPlane = false ;
      }
      else {
        msgTypePlane = baseType ;
        packPlane = true ;
      }
    }

    if (planeMin) {
      /* contiguous memory */
      if (packPlane) {
        destAddr = &domain.commDataSend[pmsg * maxPlaneComm] ;
        for (Index_t fi=0 ; fi<xferFields; ++fi) {
          Real_t *srcAddr = fieldData[fi] ;
          memcpy(destAddr, srcAddr, sendCount*sizeof(Real_t)) ;
          destAddr += sendCount ;
        }
        destAddr -= xferFields*sendCount ;
      }
      else {
        destAddr = fieldData[0] ;
      }

      MPI_Isend(destAddr, (packPlane ? xferFields*sendCount : 1),
          msgTypePlane, myRank - domain.tp()*domain.tp(), msgType,
          MPI_COMM_WORLD, &domain.sendRequest[pmsg]) ;
      ++pmsg ;
    }
    if (planeMax && doSend) {
      /* contiguous memory */
      Index_t offset = dx*dy*(dz - 1) ;
      if (packPlane) {
        destAddr = &domain.commDataSend[pmsg * maxPlaneComm] ;
        for (Index_t fi=0 ; fi<xferFields; ++fi) {
          Real_t *srcAddr = &fieldData[fi][offset] ;
          memcpy(destAddr, srcAddr, sendCount*sizeof(Real_t)) ;
          destAddr += sendCount ;
        }
        destAddr -= xferFields*sendCount ;
      }
      else {
        destAddr = &fieldData[0][offset] ;
      }

      MPI_Isend(destAddr, (packPlane ? xferFields*sendCount : 1),
          msgTypePlane, myRank + domain.tp()*domain.tp(), msgType,
          MPI_COMM_WORLD, &domain.sendRequest[pmsg]) ;
      ++pmsg ;
    }
  }
  if (rowMin | rowMax) {
    /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
    static MPI_Datatype msgTypeRow ;
    static bool packRow ;
    int sendCount = dx * dz ;

    if (msgTypeRow == 0) {
      /* Create an MPI_struct for field data */
      if (ALLOW_UNPACKED_ROW && packable) {

        static MPI_Datatype msgTypePencil ;

        /* dz pencils per plane */
        MPI_Type_vector(dz, dx, dx * dy, baseType, &msgTypePencil) ;
        MPI_Type_commit(&msgTypePencil) ;

        MPI_Type_vector(xferFields, 1, (fieldData[1] - fieldData[0]),
            msgTypePencil, &msgTypeRow) ;
        MPI_Type_commit(&msgTypeRow) ;
        packRow = false ;
      }
      else {
        msgTypeRow = baseType ;
        packRow = true ;
      }
    }

    if (rowMin) {
      /* contiguous memory */
      if (packRow) {
        destAddr = &domain.commDataSend[pmsg * maxPlaneComm] ;
        for (Index_t fi=0; fi<xferFields; ++fi) {
          Real_t *srcAddr = fieldData[fi] ;
          for (Index_t i=0; i<dz; ++i) {
            memcpy(&destAddr[i*dx], &srcAddr[i*dx*dy],
                dx*sizeof(Real_t)) ;
          }
          destAddr += sendCount ;
        }
        destAddr -= xferFields*sendCount ;
      }
      else {
        destAddr = fieldData[0] ;
      }

      MPI_Isend(destAddr, (packRow ? xferFields*sendCount : 1),
          msgTypeRow, myRank - domain.tp(), msgType,
          MPI_COMM_WORLD, &domain.sendRequest[pmsg]) ;
      ++pmsg ;
    }
    if (rowMax && doSend) {
      /* contiguous memory */
      Index_t offset = dx*(dy - 1) ;
      if (packRow) {
        destAddr = &domain.commDataSend[pmsg * maxPlaneComm] ;
        for (Index_t fi=0; fi<xferFields; ++fi) {
          Real_t *srcAddr = &fieldData[fi][offset] ;
          for (Index_t i=0; i<dz; ++i) {
            memcpy(&destAddr[i*dx], &srcAddr[i*dx*dy],
                dx*sizeof(Real_t)) ;
          }
          destAddr += sendCount ;
        }
        destAddr -= xferFields*sendCount ;
      }
      else {
        destAddr = &fieldData[0][offset] ;
      }

      MPI_Isend(destAddr, (packRow ? xferFields*sendCount : 1),
          msgTypeRow, myRank + domain.tp(), msgType,
          MPI_COMM_WORLD, &domain.sendRequest[pmsg]) ;
      ++pmsg ;
    }
  }
  if (colMin | colMax) {
    /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
    static MPI_Datatype msgTypeCol ;
    static bool packCol ;
    int sendCount = dy * dz ;

    if (msgTypeCol == 0) {
      /* Create an MPI_struct for field data */
      if (ALLOW_UNPACKED_COL && packable) {

        static MPI_Datatype msgTypePoint ;
        static MPI_Datatype msgTypePencil ;

        /* dy points per pencil */
        MPI_Type_vector(dy, 1, dx, baseType, &msgTypePoint) ;
        MPI_Type_commit(&msgTypePoint) ;

        /* dz pencils per plane */
        MPI_Type_vector(dz, 1, dx*dy, msgTypePoint, &msgTypePencil) ;
        MPI_Type_commit(&msgTypePencil) ;

        MPI_Type_vector(xferFields, 1, (fieldData[1] - fieldData[0]),
            msgTypePencil, &msgTypeCol) ;
        MPI_Type_commit(&msgTypeCol) ;
        packCol = false ;
      }
      else {
        msgTypeCol = baseType ;
        packCol = true ;
      }
    }

    if (colMin) {
      /* contiguous memory */
      if (packCol) {
        destAddr = &domain.commDataSend[pmsg * maxPlaneComm] ;
        for (Index_t fi=0; fi<xferFields; ++fi) {
          for (Index_t i=0; i<dz; ++i) {
            Real_t *srcAddr = &fieldData[fi][i*dx*dy] ;
            for (Index_t j=0; j<dy; ++j) {
              destAddr[i*dy + j] = srcAddr[j*dx] ;
            }
          }
          destAddr += sendCount ;
        }
        destAddr -= xferFields*sendCount ;
      }
      else {
        destAddr = fieldData[0] ;
      }

      MPI_Isend(destAddr, (packCol ? xferFields*sendCount : 1),
          msgTypeCol, myRank - 1, msgType,
          MPI_COMM_WORLD, &domain.sendRequest[pmsg]) ;
      ++pmsg ;
    }
    if (colMax && doSend) {
      /* contiguous memory */
      Index_t offset = dx - 1 ;
      if (packCol) {
        destAddr = &domain.commDataSend[pmsg * maxPlaneComm] ;
        for (Index_t fi=0; fi<xferFields; ++fi) {
          for (Index_t i=0; i<dz; ++i) {
            Real_t *srcAddr = &fieldData[fi][i*dx*dy + offset] ;
            for (Index_t j=0; j<dy; ++j) {
              destAddr[i*dy + j] = srcAddr[j*dx] ;
            }
          }
          destAddr += sendCount ;
        }
        destAddr -= xferFields*sendCount ;
      }
      else {
        destAddr = &fieldData[0][offset] ;
      }

      MPI_Isend(destAddr, (packCol ? xferFields*sendCount : 1),
          msgTypeCol, myRank + 1, msgType,
          MPI_COMM_WORLD, &domain.sendRequest[pmsg]) ;
      ++pmsg ;
    }
  }

  if (!planeOnly) {
    if (rowMin && colMin) {
      int toRank = myRank - domain.tp() - 1 ;
      destAddr = &domain.commDataSend[pmsg * maxPlaneComm +
        emsg * maxEdgeComm] ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
        Real_t *srcAddr = fieldData[fi] ;
        for (Index_t i=0; i<dz; ++i) {
          destAddr[i] = srcAddr[i*dx*dy] ;
        }
        destAddr += dz ;
      }
      destAddr -= xferFields*dz ;
      MPI_Isend(destAddr, xferFields*dz, baseType, toRank, msgType,
          MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
      ++emsg ;
    }

    if (rowMin && planeMin) {
      int toRank = myRank - domain.tp()*domain.tp() - domain.tp() ;
      destAddr = &domain.commDataSend[pmsg * maxPlaneComm +
        emsg * maxEdgeComm] ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
        Real_t *srcAddr = fieldData[fi] ;
        for (Index_t i=0; i<dx; ++i) {
          destAddr[i] = srcAddr[i] ;
        }
        destAddr += dx ;
      }
      destAddr -= xferFields*dx ;
      MPI_Isend(destAddr, xferFields*dx, baseType, toRank, msgType,
          MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
      ++emsg ;
    }

    if (colMin && planeMin) {
      int toRank = myRank - domain.tp()*domain.tp() - 1 ;
      destAddr = &domain.commDataSend[pmsg * maxPlaneComm +
        emsg * maxEdgeComm] ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
        Real_t *srcAddr = fieldData[fi] ;
        for (Index_t i=0; i<dy; ++i) {
          destAddr[i] = srcAddr[i*dx] ;
        }
        destAddr += dy ;
      }
      destAddr -= xferFields*dy ;
      MPI_Isend(destAddr, xferFields*dy, baseType, toRank, msgType,
          MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
      ++emsg ;
    }

    if (rowMax && colMax && doSend) {
      int toRank = myRank + domain.tp() + 1 ;
      destAddr = &domain.commDataSend[pmsg * maxPlaneComm +
        emsg * maxEdgeComm] ;
      Index_t offset = dx*dy - 1 ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
        Real_t *srcAddr = &fieldData[fi][offset] ;
        for (Index_t i=0; i<dz; ++i) {
          destAddr[i] = srcAddr[i*dx*dy] ;
        }
        destAddr += dz ;
      }
      destAddr -= xferFields*dz ;
      MPI_Isend(destAddr, xferFields*dz, baseType, toRank, msgType,
          MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
      ++emsg ;
    }

    if (rowMax && planeMax && doSend) {
      int toRank = myRank + domain.tp()*domain.tp() + domain.tp() ;
      destAddr = &domain.commDataSend[pmsg * maxPlaneComm +
        emsg * maxEdgeComm] ;
      Index_t offset = dx*(dy-1) + dx*dy*(dz-1) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
        Real_t *srcAddr = &fieldData[fi][offset] ;
        for (Index_t i=0; i<dx; ++i) {
          destAddr[i] = srcAddr[i] ;
        }
        destAddr += dx ;
      }
      destAddr -= xferFields*dx ;
      MPI_Isend(destAddr, xferFields*dx, baseType, toRank, msgType,
          MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
      ++emsg ;
    }

    if (colMax && planeMax && doSend) {
      int toRank = myRank + domain.tp()*domain.tp() + 1 ;
      destAddr = &domain.commDataSend[pmsg * maxPlaneComm +
        emsg * maxEdgeComm] ;
      Index_t offset = dx*dy*(dz-1) + dx - 1 ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
        Real_t *srcAddr = &fieldData[fi][offset] ;
        for (Index_t i=0; i<dy; ++i) {
          destAddr[i] = srcAddr[i*dx] ;
        }
        destAddr += dy ;
      }
      destAddr -= xferFields*dy ;
      MPI_Isend(destAddr, xferFields*dy, baseType, toRank, msgType,
          MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
      ++emsg ;
    }

    if (rowMax && colMin && doSend) {
      int toRank = myRank + domain.tp() - 1 ;
      destAddr = &domain.commDataSend[pmsg * maxPlaneComm +
        emsg * maxEdgeComm] ;
      Index_t offset = dx*(dy-1) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
        Real_t *srcAddr = &fieldData[fi][offset] ;
        for (Index_t i=0; i<dz; ++i) {
          destAddr[i] = srcAddr[i*dx*dy] ;
        }
        destAddr += dz ;
      }
      destAddr -= xferFields*dz ;
      MPI_Isend(destAddr, xferFields*dz, baseType, toRank, msgType,
          MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
      ++emsg ;
    }

    if (rowMin && planeMax && doSend) {
      int toRank = myRank + domain.tp()*domain.tp() - domain.tp() ;
      destAddr = &domain.commDataSend[pmsg * maxPlaneComm +
        emsg * maxEdgeComm] ;
      Index_t offset = dx*dy*(dz-1) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
        Real_t *srcAddr = &fieldData[fi][offset] ;
        for (Index_t i=0; i<dx; ++i) {
          destAddr[i] = srcAddr[i] ;
        }
        destAddr += dx ;
      }
      destAddr -= xferFields*dx ;
      MPI_Isend(destAddr, xferFields*dx, baseType, toRank, msgType,
          MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
      ++emsg ;
    }

    if (colMin && planeMax && doSend) {
      int toRank = myRank + domain.tp()*domain.tp() - 1 ;
      destAddr = &domain.commDataSend[pmsg * maxPlaneComm +
        emsg * maxEdgeComm] ;
      Index_t offset = dx*dy*(dz-1) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
        Real_t *srcAddr = &fieldData[fi][offset] ;
        for (Index_t i=0; i<dy; ++i) {
          destAddr[i] = srcAddr[i*dx] ;
        }
        destAddr += dy ;
      }
      destAddr -= xferFields*dy ;
      MPI_Isend(destAddr, xferFields*dy, baseType, toRank, msgType,
          MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
      ++emsg ;
    }

    if (rowMin && colMax) {
      int toRank = myRank - domain.tp() + 1 ;
      destAddr = &domain.commDataSend[pmsg * maxPlaneComm +
        emsg * maxEdgeComm] ;
      Index_t offset = dx - 1 ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
        Real_t *srcAddr = &fieldData[fi][offset] ;
        for (Index_t i=0; i<dz; ++i) {
          destAddr[i] = srcAddr[i*dx*dy] ;
        }
        destAddr += dz ;
      }
      destAddr -= xferFields*dz ;
      MPI_Isend(destAddr, xferFields*dz, baseType, toRank, msgType,
          MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
      ++emsg ;
    }

    if (rowMax && planeMin) {
      int toRank = myRank - domain.tp()*domain.tp() + domain.tp() ;
      destAddr = &domain.commDataSend[pmsg * maxPlaneComm +
        emsg * maxEdgeComm] ;
      Index_t offset = dx*(dy - 1) ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
        Real_t *srcAddr = &fieldData[fi][offset] ;
        for (Index_t i=0; i<dx; ++i) {
          destAddr[i] = srcAddr[i] ;
        }
        destAddr += dx ;
      }
      destAddr -= xferFields*dx ;
      MPI_Isend(destAddr, xferFields*dx, baseType, toRank, msgType,
          MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg]) ;
      ++emsg ;
    }

    if (colMax && planeMin) {
      int toRank = myRank - domain.tp()*domain.tp() + 1 ;
      destAddr = &domain.commDataSend[pmsg * maxPlaneComm +
        emsg * maxEdgeComm] ;
      Index_t offset = dx - 1 ;
      for (Index_t fi=0; fi<xferFields; ++fi) {
        Real_t *srcAddr = &fieldData[fi][offset] ;
        for (Index_t i=0; i<dy; ++i) {
          destAddr[i] = srcAddr[i*dx] ;
        }
        destAddr += dy ;
      }
      destAddr -= xferFields*dy ;
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
        comBuf[fi] = fieldData[fi][0] ;
      }
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
        comBuf[fi] = fieldData[fi][idx] ;
      }
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
        comBuf[fi] = fieldData[fi][idx] ;
      }
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
        comBuf[fi] = fieldData[fi][idx] ;
      }
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
        comBuf[fi] = fieldData[fi][idx] ;
      }
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
        comBuf[fi] = fieldData[fi][idx] ;
      }
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
        comBuf[fi] = fieldData[fi][idx] ;
      }
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
        comBuf[fi] = fieldData[fi][idx] ;
      }
      MPI_Isend(comBuf, xferFields, baseType, toRank, msgType,
          MPI_COMM_WORLD, &domain.sendRequest[pmsg+emsg+cmsg]) ;
      ++cmsg ;
    }
  }

  MPI_Waitall(26, domain.sendRequest, status) ;
}

/******************************************/

void CommSBN(Domain& domain, int xferFields, Real_t **fieldData) {
  if (domain.numRanks() == 1)
    return ;

  /* summation order should be from smallest value to largest */
  /* or we could try out kahan summation! */

  int myRank ;
  Index_t maxPlaneComm = xferFields * domain.maxPlaneSize() ;
  Index_t maxEdgeComm  = xferFields * domain.maxEdgeSize() ;
  Index_t pmsg = 0 ; /* plane comm msg */
  Index_t emsg = 0 ; /* edge comm msg */
  Index_t cmsg = 0 ; /* corner comm msg */
  Index_t dx = domain.sizeX() + 1 ;
  Index_t dy = domain.sizeY() + 1 ;
  Index_t dz = domain.sizeZ() + 1 ;
  MPI_Status status ;
  Real_t *srcAddr ;
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

  MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;

  if (planeMin | planeMax) {
    /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
    Index_t opCount = dx * dy ;

    if (planeMin) {
      /* contiguous memory */
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
      MPI_Wait(&domain.recvRequest[pmsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
        Real_t *destAddr = fieldData[fi] ;
        for (Index_t i=0; i<opCount; ++i) {
          destAddr[i] += srcAddr[i] ;
        }
        srcAddr += opCount ;
      }
      ++pmsg ;
    }
    if (planeMax) {
      /* contiguous memory */
      Index_t offset = dx*dy*(dz - 1) ;
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
      MPI_Wait(&domain.recvRequest[pmsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
        Real_t *destAddr = &fieldData[fi][offset] ;
        for (Index_t i=0; i<opCount; ++i) {
          destAddr[i] += srcAddr[i] ;
        }
        srcAddr += opCount ;
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
        for (Index_t i=0; i<dz; ++i) {
          Real_t *destAddr = &fieldData[fi][i*dx*dy] ;
          for (Index_t j=0; j<dx; ++j) {
            destAddr[j] += srcAddr[i*dx + j] ;
          }
        }
        srcAddr += opCount ;
      }
      ++pmsg ;
    }
    if (rowMax) {
      /* contiguous memory */
      Index_t offset = dx*(dy - 1) ;
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
      MPI_Wait(&domain.recvRequest[pmsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
        for (Index_t i=0; i<dz; ++i) {
          Real_t *destAddr = &fieldData[fi][offset + i*dx*dy] ;
          for (Index_t j=0; j<dx; ++j) {
            destAddr[j] += srcAddr[i*dx + j] ;
          }
        }
        srcAddr += opCount ;
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
        for (Index_t i=0; i<dz; ++i) {
          Real_t *destAddr = &fieldData[fi][i*dx*dy] ;
          for (Index_t j=0; j<dy; ++j) {
            destAddr[j*dx] += srcAddr[i*dy + j] ;
          }
        }
        srcAddr += opCount ;
      }
      ++pmsg ;
    }
    if (colMax) {
      /* contiguous memory */
      Index_t offset = dx - 1 ;
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
      MPI_Wait(&domain.recvRequest[pmsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
        for (Index_t i=0; i<dz; ++i) {
          Real_t *destAddr = &fieldData[fi][offset + i*dx*dy] ;
          for (Index_t j=0; j<dy; ++j) {
            destAddr[j*dx] += srcAddr[i*dy + j] ;
          }
        }
        srcAddr += opCount ;
      }
      ++pmsg ;
    }
  }

  if (rowMin & colMin) {
    srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm] ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
    for (Index_t fi=0 ; fi<xferFields; ++fi) {
      Real_t *destAddr = fieldData[fi] ;
      for (Index_t i=0; i<dz; ++i) {
        destAddr[i*dx*dy] += srcAddr[i] ;
      }
      srcAddr += dz ;
    }
    ++emsg ;
  }

  if (rowMin & planeMin) {
    srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm] ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
    for (Index_t fi=0 ; fi<xferFields; ++fi) {
      Real_t *destAddr = fieldData[fi] ;
      for (Index_t i=0; i<dx; ++i) {
        destAddr[i] += srcAddr[i] ;
      }
      srcAddr += dx ;
    }
    ++emsg ;
  }

  if (colMin & planeMin) {
    srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm] ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
    for (Index_t fi=0 ; fi<xferFields; ++fi) {
      Real_t *destAddr = fieldData[fi] ;
      for (Index_t i=0; i<dy; ++i) {
        destAddr[i*dx] += srcAddr[i] ;
      }
      srcAddr += dy ;
    }
    ++emsg ;
  }

  if (rowMax & colMax) {
    srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm] ;
    Index_t offset = dx*dy - 1 ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
    for (Index_t fi=0 ; fi<xferFields; ++fi) {
      Real_t *destAddr = &fieldData[fi][offset] ;
      for (Index_t i=0; i<dz; ++i) {
        destAddr[i*dx*dy] += srcAddr[i] ;
      }
      srcAddr += dz ;
    }
    ++emsg ;
  }

  if (rowMax & planeMax) {
    srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm] ;
    Index_t offset = dx*(dy-1) + dx*dy*(dz-1) ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
    for (Index_t fi=0 ; fi<xferFields; ++fi) {
      Real_t *destAddr = &fieldData[fi][offset] ;
      for (Index_t i=0; i<dx; ++i) {
        destAddr[i] += srcAddr[i] ;
      }
      srcAddr += dx ;
    }
    ++emsg ;
  }

  if (colMax & planeMax) {
    srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm] ;
    Index_t offset = dx*dy*(dz-1) + dx - 1 ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
    for (Index_t fi=0 ; fi<xferFields; ++fi) {
      Real_t *destAddr = &fieldData[fi][offset] ;
      for (Index_t i=0; i<dy; ++i) {
        destAddr[i*dx] += srcAddr[i] ;
      }
      srcAddr += dy ;
    }
    ++emsg ;
  }

  if (rowMax & colMin) {
    srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm] ;
    Index_t offset = dx*(dy-1) ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
    for (Index_t fi=0 ; fi<xferFields; ++fi) {
      Real_t *destAddr = &fieldData[fi][offset] ;
      for (Index_t i=0; i<dz; ++i) {
        destAddr[i*dx*dy] += srcAddr[i] ;
      }
      srcAddr += dz ;
    }
    ++emsg ;
  }

  if (rowMin & planeMax) {
    srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm] ;
    Index_t offset = dx*dy*(dz-1) ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
    for (Index_t fi=0 ; fi<xferFields; ++fi) {
      Real_t *destAddr = &fieldData[fi][offset] ;
      for (Index_t i=0; i<dx; ++i) {
        destAddr[i] += srcAddr[i] ;
      }
      srcAddr += dx ;
    }
    ++emsg ;
  }

  if (colMin & planeMax) {
    srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm] ;
    Index_t offset = dx*dy*(dz-1) ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
    for (Index_t fi=0 ; fi<xferFields; ++fi) {
      Real_t *destAddr = &fieldData[fi][offset] ;
      for (Index_t i=0; i<dy; ++i) {
        destAddr[i*dx] += srcAddr[i] ;
      }
      srcAddr += dy ;
    }
    ++emsg ;
  }

  if (rowMin & colMax) {
    srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm] ;
    Index_t offset = dx - 1 ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
    for (Index_t fi=0 ; fi<xferFields; ++fi) {
      Real_t *destAddr = &fieldData[fi][offset] ;
      for (Index_t i=0; i<dz; ++i) {
        destAddr[i*dx*dy] += srcAddr[i] ;
      }
      srcAddr += dz ;
    }
    ++emsg ;
  }

  if (rowMax & planeMin) {
    srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm] ;
    Index_t offset = dx*(dy - 1) ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
    for (Index_t fi=0 ; fi<xferFields; ++fi) {
      Real_t *destAddr = &fieldData[fi][offset] ;
      for (Index_t i=0; i<dx; ++i) {
        destAddr[i] += srcAddr[i] ;
      }
      srcAddr += dx ;
    }
    ++emsg ;
  }

  if (colMax & planeMin) {
    srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm] ;
    Index_t offset = dx - 1 ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
    for (Index_t fi=0 ; fi<xferFields; ++fi) {
      Real_t *destAddr = &fieldData[fi][offset] ;
      for (Index_t i=0; i<dy; ++i) {
        destAddr[i*dx] += srcAddr[i] ;
      }
      srcAddr += dy ;
    }
    ++emsg ;
  }


  if (rowMin & colMin & planeMin) {
    /* corner at domain logical coord (0, 0, 0) */
    Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm +
      cmsg * CACHE_COHERENCE_PAD_REAL] ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
    for (Index_t fi=0; fi<xferFields; ++fi) {
      fieldData[fi][0] += comBuf[fi] ;
    }
    ++cmsg ;
  }
  if (rowMin & colMin & planeMax) {
    /* corner at domain logical coord (0, 0, 1) */
    Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm +
      cmsg * CACHE_COHERENCE_PAD_REAL] ;
    Index_t idx = dx*dy*(dz - 1) ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
    for (Index_t fi=0; fi<xferFields; ++fi) {
      fieldData[fi][idx] += comBuf[fi] ;
    }
    ++cmsg ;
  }
  if (rowMin & colMax & planeMin) {
    /* corner at domain logical coord (1, 0, 0) */
    Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm +
      cmsg * CACHE_COHERENCE_PAD_REAL] ;
    Index_t idx = dx - 1 ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
    for (Index_t fi=0; fi<xferFields; ++fi) {
      fieldData[fi][idx] += comBuf[fi] ;
    }
    ++cmsg ;
  }
  if (rowMin & colMax & planeMax) {
    /* corner at domain logical coord (1, 0, 1) */
    Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm +
      cmsg * CACHE_COHERENCE_PAD_REAL] ;
    Index_t idx = dx*dy*(dz - 1) + (dx - 1) ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
    for (Index_t fi=0; fi<xferFields; ++fi) {
      fieldData[fi][idx] += comBuf[fi] ;
    }
    ++cmsg ;
  }
  if (rowMax & colMin & planeMin) {
    /* corner at domain logical coord (0, 1, 0) */
    Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm +
      cmsg * CACHE_COHERENCE_PAD_REAL] ;
    Index_t idx = dx*(dy - 1) ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
    for (Index_t fi=0; fi<xferFields; ++fi) {
      fieldData[fi][idx] += comBuf[fi] ;
    }
    ++cmsg ;
  }
  if (rowMax & colMin & planeMax) {
    /* corner at domain logical coord (0, 1, 1) */
    Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm +
      cmsg * CACHE_COHERENCE_PAD_REAL] ;
    Index_t idx = dx*dy*(dz - 1) + dx*(dy - 1) ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
    for (Index_t fi=0; fi<xferFields; ++fi) {
      fieldData[fi][idx] += comBuf[fi] ;
    }
    ++cmsg ;
  }
  if (rowMax & colMax & planeMin) {
    /* corner at domain logical coord (1, 1, 0) */
    Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm +
      cmsg * CACHE_COHERENCE_PAD_REAL] ;
    Index_t idx = dx*dy - 1 ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
    for (Index_t fi=0; fi<xferFields; ++fi) {
      fieldData[fi][idx] += comBuf[fi] ;
    }
    ++cmsg ;
  }
  if (rowMax & colMax & planeMax) {
    /* corner at domain logical coord (1, 1, 1) */
    Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm +
      cmsg * CACHE_COHERENCE_PAD_REAL] ;
    Index_t idx = dx*dy*dz - 1 ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
    for (Index_t fi=0; fi<xferFields; ++fi) {
      fieldData[fi][idx] += comBuf[fi] ;
    }
    ++cmsg ;
  }
}

/******************************************/

void CommSyncPosVel(Domain& domain) {
  if (domain.numRanks() == 1)
    return ;

  int myRank ;
  bool doRecv = false ;
  Index_t xferFields = 6 ; /* x, y, z, xd, yd, zd */
  Real_t *fieldData[6] ;
  Index_t maxPlaneComm = xferFields * domain.maxPlaneSize() ;
  Index_t maxEdgeComm  = xferFields * domain.maxEdgeSize() ;
  Index_t pmsg = 0 ; /* plane comm msg */
  Index_t emsg = 0 ; /* edge comm msg */
  Index_t cmsg = 0 ; /* corner comm msg */
  Index_t dx = domain.sizeX() + 1 ;
  Index_t dy = domain.sizeY() + 1 ;
  Index_t dz = domain.sizeZ() + 1 ;
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

  fieldData[0] = domain.x() ;
  fieldData[1] = domain.y() ;
  fieldData[2] = domain.z() ;
  fieldData[3] = domain.xd() ;
  fieldData[4] = domain.yd() ;
  fieldData[5] = domain.zd() ;

  MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;

  if (planeMin | planeMax) {
    /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
    Index_t opCount = dx * dy ;

    if (planeMin && doRecv) {
      /* contiguous memory */
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
      MPI_Wait(&domain.recvRequest[pmsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
        Real_t *destAddr = fieldData[fi] ;
        for (Index_t i=0; i<opCount; ++i) {
          destAddr[i] = srcAddr[i] ;
        }
        srcAddr += opCount ;
      }
      ++pmsg ;
    }
    if (planeMax) {
      /* contiguous memory */
      Index_t offset = dx*dy*(dz - 1) ;
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
      MPI_Wait(&domain.recvRequest[pmsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
        Real_t *destAddr = &fieldData[fi][offset] ;
        for (Index_t i=0; i<opCount; ++i) {
          destAddr[i] = srcAddr[i] ;
        }
        srcAddr += opCount ;
      }
      ++pmsg ;
    }
  }

  if (rowMin | rowMax) {
    /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
    Index_t opCount = dx * dz ;

    if (rowMin && doRecv) {
      /* contiguous memory */
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
      MPI_Wait(&domain.recvRequest[pmsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
        for (Index_t i=0; i<dz; ++i) {
          Real_t *destAddr = &fieldData[fi][i*dx*dy] ;
          for (Index_t j=0; j<dx; ++j) {
            destAddr[j] = srcAddr[i*dx + j] ;
          }
        }
        srcAddr += opCount ;
      }
      ++pmsg ;
    }
    if (rowMax) {
      /* contiguous memory */
      Index_t offset = dx*(dy - 1) ;
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
      MPI_Wait(&domain.recvRequest[pmsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
        for (Index_t i=0; i<dz; ++i) {
          Real_t *destAddr = &fieldData[fi][offset + i*dx*dy] ;
          for (Index_t j=0; j<dx; ++j) {
            destAddr[j] = srcAddr[i*dx + j] ;
          }
        }
        srcAddr += opCount ;
      }
      ++pmsg ;
    }
  }
  if (colMin | colMax) {
    /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
    Index_t opCount = dy * dz ;

    if (colMin && doRecv) {
      /* contiguous memory */
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
      MPI_Wait(&domain.recvRequest[pmsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
        for (Index_t i=0; i<dz; ++i) {
          Real_t *destAddr = &fieldData[fi][i*dx*dy] ;
          for (Index_t j=0; j<dy; ++j) {
            destAddr[j*dx] = srcAddr[i*dy + j] ;
          }
        }
        srcAddr += opCount ;
      }
      ++pmsg ;
    }
    if (colMax) {
      /* contiguous memory */
      Index_t offset = dx - 1 ;
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
      MPI_Wait(&domain.recvRequest[pmsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
        for (Index_t i=0; i<dz; ++i) {
          Real_t *destAddr = &fieldData[fi][offset + i*dx*dy] ;
          for (Index_t j=0; j<dy; ++j) {
            destAddr[j*dx] = srcAddr[i*dy + j] ;
          }
        }
        srcAddr += opCount ;
      }
      ++pmsg ;
    }
  }

  if (rowMin && colMin && doRecv) {
    srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm] ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
    for (Index_t fi=0 ; fi<xferFields; ++fi) {
      Real_t *destAddr = fieldData[fi] ;
      for (Index_t i=0; i<dz; ++i) {
        destAddr[i*dx*dy] = srcAddr[i] ;
      }
      srcAddr += dz ;
    }
    ++emsg ;
  }

  if (rowMin && planeMin && doRecv) {
    srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm] ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
    for (Index_t fi=0 ; fi<xferFields; ++fi) {
      Real_t *destAddr = fieldData[fi] ;
      for (Index_t i=0; i<dx; ++i) {
        destAddr[i] = srcAddr[i] ;
      }
      srcAddr += dx ;
    }
    ++emsg ;
  }

  if (colMin && planeMin && doRecv) {
    srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm] ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
    for (Index_t fi=0 ; fi<xferFields; ++fi) {
      Real_t *destAddr = fieldData[fi] ;
      for (Index_t i=0; i<dy; ++i) {
        destAddr[i*dx] = srcAddr[i] ;
      }
      srcAddr += dy ;
    }
    ++emsg ;
  }

  if (rowMax && colMax) {
    srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm] ;
    Index_t offset = dx*dy - 1 ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
    for (Index_t fi=0 ; fi<xferFields; ++fi) {
      Real_t *destAddr = &fieldData[fi][offset] ;
      for (Index_t i=0; i<dz; ++i) {
        destAddr[i*dx*dy] = srcAddr[i] ;
      }
      srcAddr += dz ;
    }
    ++emsg ;
  }

  if (rowMax && planeMax) {
    srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm] ;
    Index_t offset = dx*(dy-1) + dx*dy*(dz-1) ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
    for (Index_t fi=0 ; fi<xferFields; ++fi) {
      Real_t *destAddr = &fieldData[fi][offset] ;
      for (Index_t i=0; i<dx; ++i) {
        destAddr[i] = srcAddr[i] ;
      }
      srcAddr += dx ;
    }
    ++emsg ;
  }

  if (colMax && planeMax) {
    srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm] ;
    Index_t offset = dx*dy*(dz-1) + dx - 1 ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
    for (Index_t fi=0 ; fi<xferFields; ++fi) {
      Real_t *destAddr = &fieldData[fi][offset] ;
      for (Index_t i=0; i<dy; ++i) {
        destAddr[i*dx] = srcAddr[i] ;
      }
      srcAddr += dy ;
    }
    ++emsg ;
  }

  if (rowMax && colMin) {
    srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm] ;
    Index_t offset = dx*(dy-1) ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
    for (Index_t fi=0 ; fi<xferFields; ++fi) {
      Real_t *destAddr = &fieldData[fi][offset] ;
      for (Index_t i=0; i<dz; ++i) {
        destAddr[i*dx*dy] = srcAddr[i] ;
      }
      srcAddr += dz ;
    }
    ++emsg ;
  }

  if (rowMin && planeMax) {
    srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm] ;
    Index_t offset = dx*dy*(dz-1) ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
    for (Index_t fi=0 ; fi<xferFields; ++fi) {
      Real_t *destAddr = &fieldData[fi][offset] ;
      for (Index_t i=0; i<dx; ++i) {
        destAddr[i] = srcAddr[i] ;
      }
      srcAddr += dx ;
    }
    ++emsg ;
  }

  if (colMin && planeMax) {
    srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm] ;
    Index_t offset = dx*dy*(dz-1) ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
    for (Index_t fi=0 ; fi<xferFields; ++fi) {
      Real_t *destAddr = &fieldData[fi][offset] ;
      for (Index_t i=0; i<dy; ++i) {
        destAddr[i*dx] = srcAddr[i] ;
      }
      srcAddr += dy ;
    }
    ++emsg ;
  }

  if (rowMin && colMax && doRecv) {
    srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm] ;
    Index_t offset = dx - 1 ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
    for (Index_t fi=0 ; fi<xferFields; ++fi) {
      Real_t *destAddr = &fieldData[fi][offset] ;
      for (Index_t i=0; i<dz; ++i) {
        destAddr[i*dx*dy] = srcAddr[i] ;
      }
      srcAddr += dz ;
    }
    ++emsg ;
  }

  if (rowMax && planeMin && doRecv) {
    srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm] ;
    Index_t offset = dx*(dy - 1) ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
    for (Index_t fi=0 ; fi<xferFields; ++fi) {
      Real_t *destAddr = &fieldData[fi][offset] ;
      for (Index_t i=0; i<dx; ++i) {
        destAddr[i] = srcAddr[i] ;
      }
      srcAddr += dx ;
    }
    ++emsg ;
  }

  if (colMax && planeMin && doRecv) {
    srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm] ;
    Index_t offset = dx - 1 ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg], &status) ;
    for (Index_t fi=0 ; fi<xferFields; ++fi) {
      Real_t *destAddr = &fieldData[fi][offset] ;
      for (Index_t i=0; i<dy; ++i) {
        destAddr[i*dx] = srcAddr[i] ;
      }
      srcAddr += dy ;
    }
    ++emsg ;
  }


  if (rowMin && colMin && planeMin && doRecv) {
    /* corner at domain logical coord (0, 0, 0) */
    Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm +
      cmsg * CACHE_COHERENCE_PAD_REAL] ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
    for (Index_t fi=0; fi<xferFields; ++fi) {
      fieldData[fi][0] = comBuf[fi] ;
    }
    ++cmsg ;
  }
  if (rowMin && colMin && planeMax) {
    /* corner at domain logical coord (0, 0, 1) */
    Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm +
      cmsg * CACHE_COHERENCE_PAD_REAL] ;
    Index_t idx = dx*dy*(dz - 1) ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
    for (Index_t fi=0; fi<xferFields; ++fi) {
      fieldData[fi][idx] = comBuf[fi] ;
    }
    ++cmsg ;
  }
  if (rowMin && colMax && planeMin && doRecv) {
    /* corner at domain logical coord (1, 0, 0) */
    Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm +
      cmsg * CACHE_COHERENCE_PAD_REAL] ;
    Index_t idx = dx - 1 ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
    for (Index_t fi=0; fi<xferFields; ++fi) {
      fieldData[fi][idx] = comBuf[fi] ;
    }
    ++cmsg ;
  }
  if (rowMin && colMax && planeMax) {
    /* corner at domain logical coord (1, 0, 1) */
    Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm +
      cmsg * CACHE_COHERENCE_PAD_REAL] ;
    Index_t idx = dx*dy*(dz - 1) + (dx - 1) ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
    for (Index_t fi=0; fi<xferFields; ++fi) {
      fieldData[fi][idx] = comBuf[fi] ;
    }
    ++cmsg ;
  }
  if (rowMax && colMin && planeMin && doRecv) {
    /* corner at domain logical coord (0, 1, 0) */
    Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm +
      cmsg * CACHE_COHERENCE_PAD_REAL] ;
    Index_t idx = dx*(dy - 1) ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
    for (Index_t fi=0; fi<xferFields; ++fi) {
      fieldData[fi][idx] = comBuf[fi] ;
    }
    ++cmsg ;
  }
  if (rowMax && colMin && planeMax) {
    /* corner at domain logical coord (0, 1, 1) */
    Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm +
      cmsg * CACHE_COHERENCE_PAD_REAL] ;
    Index_t idx = dx*dy*(dz - 1) + dx*(dy - 1) ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
    for (Index_t fi=0; fi<xferFields; ++fi) {
      fieldData[fi][idx] = comBuf[fi] ;
    }
    ++cmsg ;
  }
  if (rowMax && colMax && planeMin && doRecv) {
    /* corner at domain logical coord (1, 1, 0) */
    Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm +
      cmsg * CACHE_COHERENCE_PAD_REAL] ;
    Index_t idx = dx*dy - 1 ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
    for (Index_t fi=0; fi<xferFields; ++fi) {
      fieldData[fi][idx] = comBuf[fi] ;
    }
    ++cmsg ;
  }
  if (rowMax && colMax && planeMax) {
    /* corner at domain logical coord (1, 1, 1) */
    Real_t *comBuf = &domain.commDataRecv[pmsg * maxPlaneComm +
      emsg * maxEdgeComm +
      cmsg * CACHE_COHERENCE_PAD_REAL] ;
    Index_t idx = dx*dy*dz - 1 ;
    MPI_Wait(&domain.recvRequest[pmsg+emsg+cmsg], &status) ;
    for (Index_t fi=0; fi<xferFields; ++fi) {
      fieldData[fi][idx] = comBuf[fi] ;
    }
    ++cmsg ;
  }
}

/******************************************/

void CommMonoQ(Domain& domain)
{
  if (domain.numRanks() == 1)
    return ;

  int myRank ;
  Index_t xferFields = 3 ; /* delv_xi, delv_eta, delv_zeta */
  Real_t *fieldData[3] ;
  Index_t maxPlaneComm = xferFields * domain.maxPlaneSize() ;
  Index_t pmsg = 0 ; /* plane comm msg */
  Index_t dx = domain.sizeX() ;
  Index_t dy = domain.sizeY() ;
  Index_t dz = domain.sizeZ() ;
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
  fieldData[0] = &(domain.delv_xi(domain.numElem())) ;
  fieldData[1] = &(domain.delv_eta(domain.numElem())) ;
  fieldData[2] = &(domain.delv_zeta(domain.numElem())) ;

  MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;

  if (planeMin | planeMax) {
    /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
    Index_t opCount = dx * dy ;

    if (planeMin) {
      /* contiguous memory */
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
      MPI_Wait(&domain.recvRequest[pmsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
        Real_t *destAddr = fieldData[fi] ;
        for (Index_t i=0; i<opCount; ++i) {
          destAddr[i] = srcAddr[i] ;
        }
        srcAddr += opCount ;
        fieldData[fi] += opCount ;
      }
      ++pmsg ;
    }
    if (planeMax) {
      /* contiguous memory */
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
      MPI_Wait(&domain.recvRequest[pmsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
        Real_t *destAddr = fieldData[fi] ;
        for (Index_t i=0; i<opCount; ++i) {
          destAddr[i] = srcAddr[i] ;
        }
        srcAddr += opCount ;
        fieldData[fi] += opCount ;
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
        Real_t *destAddr = fieldData[fi] ;
        for (Index_t i=0; i<opCount; ++i) {
          destAddr[i] = srcAddr[i] ;
        }
        srcAddr += opCount ;
        fieldData[fi] += opCount ;
      }
      ++pmsg ;
    }
    if (rowMax) {
      /* contiguous memory */
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
      MPI_Wait(&domain.recvRequest[pmsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
        Real_t *destAddr = fieldData[fi] ;
        for (Index_t i=0; i<opCount; ++i) {
          destAddr[i] = srcAddr[i] ;
        }
        srcAddr += opCount ;
        fieldData[fi] += opCount ;
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
        Real_t *destAddr = fieldData[fi] ;
        for (Index_t i=0; i<opCount; ++i) {
          destAddr[i] = srcAddr[i] ;
        }
        srcAddr += opCount ;
        fieldData[fi] += opCount ;
      }
      ++pmsg ;
    }
    if (colMax) {
      /* contiguous memory */
      srcAddr = &domain.commDataRecv[pmsg * maxPlaneComm] ;
      MPI_Wait(&domain.recvRequest[pmsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
        Real_t *destAddr = fieldData[fi] ;
        for (Index_t i=0; i<opCount; ++i) {
          destAddr[i] = srcAddr[i] ;
        }
        srcAddr += opCount ;
      }
      ++pmsg ;
    }
  }
}

#endif
