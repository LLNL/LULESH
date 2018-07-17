#include <math.h>
#if USE_MPI
# include <mpi.h>
#endif
#if _OPENMP
#include <omp.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <cstdlib>

#include "lulesh.h"

/////////////////////////////////////////////////////////////////////
Domain::Domain(Int_t numRanks, Index_t colLoc,
               Index_t rowLoc, Index_t planeLoc,
               Index_t nx, int tp, int nr, int balance, Int_t cost)
  :
  m_e_cut(Real_t(1.0e-7)),
  m_p_cut(Real_t(1.0e-7)),
  m_q_cut(Real_t(1.0e-7)),
  m_v_cut(Real_t(1.0e-10)),
  m_u_cut(Real_t(1.0e-7)),
  m_hgcoef(Real_t(3.0)),
  m_ss4o3(Real_t(4.0)/Real_t(3.0)),
  m_qstop(Real_t(1.0e+12)),
  m_monoq_max_slope(Real_t(1.0)),
  m_monoq_limiter_mult(Real_t(2.0)),
  m_qlc_monoq(Real_t(0.5)),
  m_qqc_monoq(Real_t(2.0)/Real_t(3.0)),
  m_qqc(Real_t(2.0)),
  m_eosvmax(Real_t(1.0e+9)),
  m_eosvmin(Real_t(1.0e-9)),
  m_pmin(Real_t(0.)),
  m_emin(Real_t(-1.0e+15)),
  m_dvovmax(Real_t(0.1)),
  m_refdens(Real_t(1.0))
{

  Index_t edgeElems = nx ;
  Index_t edgeNodes = edgeElems+1 ;
  this->cost() = cost;

  m_tp       = tp ;
  m_numRanks = numRanks ;

  ///////////////////////////////
  //   Initialize Sedov Mesh
  ///////////////////////////////

  // construct a uniform box for this processor

  m_colLoc   =   colLoc ;
  m_rowLoc   =   rowLoc ;
  m_planeLoc = planeLoc ;

  m_sizeX = edgeElems ;
  m_sizeY = edgeElems ;
  m_sizeZ = edgeElems ;
  m_numElem = edgeElems*edgeElems*edgeElems ;

  m_numNode = edgeNodes*edgeNodes*edgeNodes ;

  m_regNumList = new Index_t[m_numElem] ;  // material indexset

  m_nodelist.resize(8*m_numElem);

  // elem connectivities through face 
  m_lxim.resize(m_numElem);
  m_lxip.resize(m_numElem);
  m_letam.resize(m_numElem);
  m_letap.resize(m_numElem);
  m_lzetam.resize(m_numElem);
  m_lzetap.resize(m_numElem);

  m_elemBC.resize(m_numElem);

  m_e.resize(m_numElem);
  m_p.resize(m_numElem);

  m_q.resize(m_numElem);
  m_ql.resize(m_numElem);
  m_qq.resize(m_numElem);

  m_v.resize(m_numElem);

  m_volo.resize(m_numElem);
  m_delv.resize(m_numElem);
  m_vdov.resize(m_numElem);

  m_arealg.resize(m_numElem);

  m_ss.resize(m_numElem);

  m_elemMass.resize(m_numElem);

  // Node-centered 

  m_x.resize(m_numNode);  // coordinates 
  m_y.resize(m_numNode);
  m_z.resize(m_numNode);

  m_xd.resize(m_numNode); // velocities 
  m_yd.resize(m_numNode);
  m_zd.resize(m_numNode);

  m_xdd.resize(m_numNode); // accelerations 
  m_ydd.resize(m_numNode);
  m_zdd.resize(m_numNode);

  m_fx.resize(m_numNode);  // forces 
  m_fy.resize(m_numNode);
  m_fz.resize(m_numNode);

  // Allocate tmp arrays
  m_fx_elem.resize(m_numElem * 8);
  m_fy_elem.resize(m_numElem * 8);
  m_fz_elem.resize(m_numElem * 8);
  m_dvdx.resize(m_numElem * 8);
  m_dvdy.resize(m_numElem * 8);
  m_dvdz.resize(m_numElem * 8);
  m_x8n.resize(m_numElem * 8);
  m_y8n.resize(m_numElem * 8);
  m_z8n.resize(m_numElem * 8);
  m_sigxx.resize(m_numElem);
  m_sigyy.resize(m_numElem);
  m_sigzz.resize(m_numElem);
  m_determ.resize(m_numElem);
  m_dxx.resize(m_numElem) ; 
  m_dyy.resize(m_numElem) ;
  m_dzz.resize(m_numElem) ;
  m_vnew.resize(m_numElem) ;
  Index_t allElem = m_numElem +  /* local elem */
    2*sizeX()*sizeY() + /* plane ghosts */
    2*sizeX()*sizeZ() + /* row ghosts */
    2*sizeY()*sizeZ() ; /* col ghosts */
  AllocateGradients(allElem);

  m_nodalMass.resize(m_numNode);  // mass 

  SetupCommBuffers(edgeNodes);

  // Basic Field Initialization 
  std::fill( m_e.begin(), m_e.end(), Real_t(0.0) );
  std::fill( m_p.begin(), m_p.end(), Real_t(0.0) );
  std::fill( m_q.begin(), m_q.end(), Real_t(0.0) );
  std::fill( m_ss.begin(), m_ss.end(), Real_t(0.0) );

  // Note - v initializes to 1.0, not 0.0!
  std::fill( m_v.begin(), m_v.end(), Real_t(1.0) );

  std::fill( m_xd.begin(), m_xd.end(), Real_t(0.0) );
  std::fill( m_yd.begin(), m_yd.end(), Real_t(0.0) );
  std::fill( m_zd.begin(), m_zd.end(), Real_t(0.0) );

  std::fill( m_xdd.begin(), m_xdd.end(), Real_t(0.0) );
  std::fill( m_ydd.begin(), m_ydd.end(), Real_t(0.0) );
  std::fill( m_zdd.begin(), m_zdd.end(), Real_t(0.0) );

  std::fill( m_nodalMass.begin(), m_nodalMass.end(), Real_t(0.0) );

  // OpenACC - init device ptrs
#ifdef _OPENACC
  m_numDevs = acc_get_num_devices(acc_device_nvidia);
#if USE_MPI
  if(m_numDevs > 1) {
    //printf("%d Nvidia accelerators found.\n\n", m_numDevs);
    int numRanks;
    int myRank;
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks) ;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;

    /* Explicitly set device number if using MPI and >1 device */
    if(numRanks > 1) {
      acc_set_device_num((myRank % m_numDevs) + 1, acc_device_nvidia);
    }
  }
#endif
#endif

  BuildMesh(nx, edgeNodes, edgeElems);

#if _OPENMP
  SetupThreadSupportStructures();
#else
  // These arrays are not used if we're not threaded
  m_nodeElemStart = NULL;
  m_nodeElemCount = NULL;
  m_nodeElemCornerList = NULL;
#endif

  // Setup region index sets. For now, these are constant sized
  // throughout the run, but could be changed every cycle to 
  // simulate effects of ALE on the lagrange solver
  CreateRegionIndexSets(nr, balance);

  // Setup symmetry nodesets
  SetupSymmetryPlanes(edgeNodes);

  // Setup element connectivities
  SetupElementConnectivities(edgeElems);

  // Setup symmetry planes and free surface boundary arrays
  SetupBoundaryConditions(edgeElems);


  // Setup defaults

  // These can be changed (requires recompile) if you want to run
  // with a fixed timestep, or to a different end time, but it's
  // probably easier/better to just run a fixed number of timesteps
  // using the -i flag in 2.x

  //dtfixed() = Real_t(5.0e-10) ; // Negative means use courant condition
  dtfixed() = Real_t(-1.0e-6) ; // Negative means use courant condition
  stoptime()  = Real_t(1.0e-2); // *Real_t(edgeElems*tp/45.0) ;

  // Initial conditions
  deltatimemultlb() = Real_t(1.1) ;
  deltatimemultub() = Real_t(1.2) ;
  dtcourant() = Real_t(1.0e+20) ;
  dthydro()   = Real_t(1.0e+20) ;
  dtmax()     = Real_t(1.0e-2) ;
  time()    = Real_t(0.) ;
  cycle()   = 0 ;

  // initialize field data 
  for (Index_t i=0; i<m_numElem; ++i) {
    Real_t x_local[8], y_local[8], z_local[8] ;
    Index_t *elemToNode = &m_nodelist[8*i] ;
    for( Index_t lnode=0 ; lnode<8 ; ++lnode )
    {
      Index_t gnode = elemToNode[lnode];
      x_local[lnode] = m_x[gnode];
      y_local[lnode] = m_y[gnode];
      z_local[lnode] = m_z[gnode];
    }

    // volume calculations
    Real_t volume = CalcElemVolume(x_local, y_local, z_local );
    m_volo[i] = volume ;
    m_elemMass[i] = volume ;
    for (Index_t j=0; j<8; ++j) {
      Index_t idx = elemToNode[j] ;
      m_nodalMass[idx] += volume / Real_t(8.0) ;
    }
  }

  // deposit initial energy
  // An energy of 3.948746e+7 is correct for a problem with
  // 45 zones along a side - we need to scale it
  const Real_t ebase = 3.948746e+7;
  Real_t scale = (nx*m_tp)/45.0;
  Real_t einit = ebase*scale*scale*scale;
  if (m_rowLoc + m_colLoc + m_planeLoc == 0) {
    // Dump into the first zone (which we know is in the corner)
    // of the domain that sits at the origin
    this->e(0) = einit;
  }

  // Initialize deltatime
  if(dtfixed() > Real_t(0.)) {
    deltatime() = dtfixed();
    printf("Using fixed timestep of %12.6e\n\n", deltatime());
  }
  else {
    //set initial deltatime base on analytic CFL calculation
    deltatime() = (.5*cbrt(m_volo[0]))/sqrt(2*einit);
  }

} // End constructor


////////////////////////////////////////////////////////////////////////////////
void
Domain::BuildMesh(Int_t nx, Int_t edgeNodes, Int_t edgeElems)
{
  Index_t meshEdgeElems = m_tp*nx ;

  // initialize nodal coordinates 
  Int_t nidx = 0 ;
  Real_t tz = Real_t(1.125)*Real_t(m_planeLoc*nx)/Real_t(meshEdgeElems) ;
  for (Index_t plane=0; plane<edgeNodes; ++plane) {
     Real_t ty = Real_t(1.125)*Real_t(m_rowLoc*nx)/Real_t(meshEdgeElems) ;
     for (Index_t row=0; row<edgeNodes; ++row) {
        Real_t tx = Real_t(1.125)*Real_t(m_colLoc*nx)/Real_t(meshEdgeElems) ;
        for (Index_t col=0; col<edgeNodes; ++col) {
           m_x[nidx] = tx ;
           m_y[nidx] = ty ;
           m_z[nidx] = tz ;
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


  // embed hexehedral elements in nodal point lattice 
  Int_t zidx = 0 ;
  nidx = 0 ;
  for (Index_t plane=0; plane<edgeElems; ++plane) {
    for (Index_t row=0; row<edgeElems; ++row) {
      for (Index_t col=0; col<edgeElems; ++col) {
	Index_t *localNode = &m_nodelist[8*zidx] ;
	localNode[0] = nidx                                       ;
	localNode[1] = nidx                                   + 1 ;
	localNode[2] = nidx                       + edgeNodes + 1 ;
	localNode[3] = nidx                       + edgeNodes     ;
	localNode[4] = nidx + edgeNodes*edgeNodes                 ;
	localNode[5] = nidx + edgeNodes*edgeNodes             + 1 ;
	localNode[6] = nidx + edgeNodes*edgeNodes + edgeNodes + 1 ;
	localNode[7] = nidx + edgeNodes*edgeNodes + edgeNodes     ;
	++zidx ;
	++nidx ;
      }
      ++nidx ;
    }
    nidx += edgeNodes ;
  }
}


////////////////////////////////////////////////////////////////////////////////
void
Domain::SetupThreadSupportStructures()
{
#if _OPENMP
   Index_t numthreads = omp_get_max_threads();
#else
   Index_t numthreads = 1;
#endif

   // These structures are always needed if using OpenACC, so just always
   // allocate them
   if (1 /*numthreads > 1*/) {
     // set up node-centered indexing of elements 
     m_nodeElemCount = new Index_t[m_numNode] ;

     for (Index_t i=0; i<m_numNode; ++i) {
       m_nodeElemCount[i] = 0 ;
     }

     for (Index_t i=0; i<m_numElem; ++i) {
       Index_t *nl = &m_nodelist[8*i] ;
       for (Index_t j=0; j < 8; ++j) {
         ++(m_nodeElemCount[nl[j]] );
       }
     }

     m_nodeElemStart = new Index_t[m_numNode] ;

     m_nodeElemStart[0] = 0;

     for (Index_t i=1; i < m_numNode; ++i) {
       m_nodeElemStart[i] =
         m_nodeElemStart[i-1] + m_nodeElemCount[i-1] ;
     }


     m_nodeElemCornerList =
       new Index_t[m_nodeElemStart[m_numNode-1] +
       m_nodeElemCount[m_numNode-1] ];

     for (Index_t i=0; i < m_numNode; ++i) {
       m_nodeElemCount[i] = 0;
     }

     for (Index_t i=0; i < m_numElem; ++i) {
       Index_t *nl = &m_nodelist[i*8] ;
       for (Index_t j=0; j < 8; ++j) {
         Index_t m = nl[j];
         Index_t k = i*8 + j ;
         Index_t offset = m_nodeElemStart[m] +
           m_nodeElemCount[m] ;
         m_nodeElemCornerList[offset] = k;
         ++(m_nodeElemCount[m]) ;
       }
     }

     Index_t clSize = m_nodeElemStart[m_numNode-1] +
       m_nodeElemCount[m_numNode-1] ;
     for (Index_t i=0; i < clSize; ++i) {
       Index_t clv = m_nodeElemCornerList[i] ;
       if ((clv < 0) || (clv > m_numElem*8)) {
         fprintf(stderr,
             "AllocateNodeElemIndexes(): nodeElemCornerList entry out of range!\n");
#if USE_MPI
         MPI_Abort(MPI_COMM_WORLD, -1);
#else
         exit(-1);
#endif
       }
     }
   }
   else {
     // These arrays are not used if we're not threaded
     m_nodeElemStart = NULL;
     m_nodeElemCount = NULL;
     m_nodeElemCornerList = NULL;
   }
}


////////////////////////////////////////////////////////////////////////////////
void
Domain::SetupCommBuffers(Int_t edgeNodes)
{
  // allocate a buffer large enough for nodal ghost data 
  Index_t maxEdgeSize = MAX(this->sizeX(), MAX(this->sizeY(), this->sizeZ()))+1 ;
  m_maxPlaneSize = CACHE_ALIGN_REAL(maxEdgeSize*maxEdgeSize) ;
  m_maxEdgeSize = CACHE_ALIGN_REAL(maxEdgeSize) ;

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
    m_maxPlaneSize * MAX_FIELDS_PER_MPI_COMM ;

  // account for edge communication 
  comBufSize +=
    ((m_rowMin & m_colMin) + (m_rowMin & m_planeMin) + (m_colMin & m_planeMin) +
     (m_rowMax & m_colMax) + (m_rowMax & m_planeMax) + (m_colMax & m_planeMax) +
     (m_rowMax & m_colMin) + (m_rowMin & m_planeMax) + (m_colMin & m_planeMax) +
     (m_rowMin & m_colMax) + (m_rowMax & m_planeMin) + (m_colMax & m_planeMin)) *
    m_maxPlaneSize * MAX_FIELDS_PER_MPI_COMM ;

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
  // prevent floating point exceptions 
  memset(this->commDataSend, 0, comBufSize*sizeof(Real_t)) ;
  memset(this->commDataRecv, 0, comBufSize*sizeof(Real_t)) ;
#endif   

  // Boundary nodesets
  if (m_colLoc == 0)
    m_symmX.resize(edgeNodes*edgeNodes);
  if (m_rowLoc == 0)
    m_symmY.resize(edgeNodes*edgeNodes);
  if (m_planeLoc == 0)
    m_symmZ.resize(edgeNodes*edgeNodes);
}


////////////////////////////////////////////////////////////////////////////////
void
Domain::CreateRegionIndexSets(Int_t nr, Int_t balance)
{
#if USE_MPI   
  Index_t myRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;
  srand(myRank);
#else
  srand(0);
  Index_t myRank = 0;
#endif
  this->numReg() = nr;
  m_regElemSize = new Int_t[numReg()];
  m_regElemlist = new Int_t*[numReg()];
  Index_t nextIndex = 0;
  //if we only have one region just fill it
  // Fill out the regNumList with material numbers, which are always
  // the region index plus one 
  if(numReg() == 1) {
    while (nextIndex < m_numElem) {
      this->regNumList(nextIndex) = 1;
      nextIndex++;
    }
    regElemSize(0) = 0;
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
    Int_t* regBinEnd = new Int_t[numReg()];
    //Determine the relative weights of all the regions.
    for (Index_t i=0 ; i<numReg() ; ++i) {
      regElemSize(i) = 0;
      costDenominator += pow((i+1), balance);  //Total cost of all regions
      regBinEnd[i] = costDenominator;  //Chance of hitting a given region is (regBinEnd[i] - regBinEdn[i-1])/costDenominator
    }
    //Until all elements are assigned
    while (nextIndex < m_numElem) {
      //pick the region
      regionVar = rand() % costDenominator;
      Index_t i = 0;
      while(regionVar >= regBinEnd[i])
        i++;
      //rotate the regions based on MPI rank.  Rotation is Rank % NumRegions
      regionNum = ((i + myRank) % numReg()) + 1;
      // make sure we don't pick the same region twice in a row
      while(regionNum == lastReg) {
        regionVar = rand() % costDenominator;
        i = 0;
        while(regionVar >= regBinEnd[i])
          i++;
        regionNum = ((i + myRank) % numReg()) + 1;
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
      while (nextIndex < runto && nextIndex < m_numElem) {
        this->regNumList(nextIndex) = regionNum;
        nextIndex++;
      }
      lastReg = regionNum;
    } 
  }
  // Convert regNumList to region index sets
  // First, count size of each region 
  for (Index_t i=0 ; i<m_numElem ; ++i) {
    int r = this->regNumList(i)-1; // region index == regnum-1
    regElemSize(r)++;
  }
  // Second, allocate each region index set
  for (Index_t i=0 ; i<numReg() ; ++i) {
    m_regElemlist[i] = new Index_t[regElemSize(i)];
    regElemSize(i) = 0;
  }
  // Third, fill index sets
  for (Index_t i=0 ; i<m_numElem ; ++i) {
    Index_t r = regNumList(i)-1;       // region index == regnum-1
    Index_t regndx = regElemSize(r)++; // Note increment
    regElemlist(r,regndx) = i;
  }

}

/////////////////////////////////////////////////////////////
void 
Domain::SetupSymmetryPlanes(Int_t edgeNodes)
{
  Int_t nidx = 0 ;
  for (Index_t i=0; i<edgeNodes; ++i) {
    Index_t planeInc = i*edgeNodes*edgeNodes ;
    Index_t rowInc   = i*edgeNodes ;
    for (Index_t j=0; j<edgeNodes; ++j) {
      if (m_planeLoc == 0) {
        m_symmZ[nidx] = rowInc   + j ;
      }
      if (m_rowLoc == 0) {
        m_symmY[nidx] = planeInc + j ;
      }
      if (m_colLoc == 0) {
        m_symmX[nidx] = planeInc + j*edgeNodes ;
      }
      ++nidx ;
    }
  }
}



/////////////////////////////////////////////////////////////
void
Domain::SetupElementConnectivities(Int_t edgeElems)
{
  m_lxim[0] = 0 ;
  for (Index_t i=1; i<m_numElem; ++i) {
    m_lxim[i]   = i-1 ;
    m_lxip[i-1] = i ;
  }
  m_lxip[m_numElem-1] = m_numElem-1 ;

  for (Index_t i=0; i<edgeElems; ++i) {
    m_letam[i] = i ; 
    m_letap[m_numElem-edgeElems+i] = m_numElem-edgeElems+i ;
  }
  for (Index_t i=edgeElems; i<m_numElem; ++i) {
    m_letam[i] = i-edgeElems ;
    m_letap[i-edgeElems] = i ;
  }

  for (Index_t i=0; i<edgeElems*edgeElems; ++i) {
    m_lzetam[i] = i ;
    m_lzetap[m_numElem-edgeElems*edgeElems+i] = m_numElem-edgeElems*edgeElems+i ;
  }
  for (Index_t i=edgeElems*edgeElems; i<m_numElem; ++i) {
    m_lzetam[i] = i - edgeElems*edgeElems ;
    m_lzetap[i-edgeElems*edgeElems] = i ;
  }
}

/////////////////////////////////////////////////////////////
void
Domain::SetupBoundaryConditions(Int_t edgeElems) 
{
  Index_t ghostIdx[6] ;  // offsets to ghost locations

  // set up boundary condition information
  std::fill(m_elemBC.begin(), m_elemBC.end(), 0);

  for (Index_t i=0; i<6; ++i) {
    ghostIdx[i] = INT_MIN ;
  }

  Int_t pidx = m_numElem ;
  if (m_planeMin != 0) {
    ghostIdx[0] = pidx ;
    pidx += sizeX()*sizeY() ;
  }

  if (m_planeMax != 0) {
    ghostIdx[1] = pidx ;
    pidx += sizeX()*sizeY() ;
  }

  if (m_rowMin != 0) {
    ghostIdx[2] = pidx ;
    pidx += sizeX()*sizeZ() ;
  }

  if (m_rowMax != 0) {
    ghostIdx[3] = pidx ;
    pidx += sizeX()*sizeZ() ;
  }

  if (m_colMin != 0) {
    ghostIdx[4] = pidx ;
    pidx += sizeY()*sizeZ() ;
  }

  if (m_colMax != 0) {
    ghostIdx[5] = pidx ;
  }

  // symmetry plane or free surface BCs 
  for (Index_t i=0; i<edgeElems; ++i) {
    Index_t planeInc = i*edgeElems*edgeElems ;
    Index_t rowInc   = i*edgeElems ;
    for (Index_t j=0; j<edgeElems; ++j) {
      if (m_planeLoc == 0) {
        m_elemBC[rowInc+j] |= ZETA_M_SYMM ;
      }
      else {
        m_elemBC[rowInc+j] |= ZETA_M_COMM ;
        m_lzetam[rowInc+j] = ghostIdx[0] + rowInc + j ;
      }

      if (m_planeLoc == m_tp-1) {
        m_elemBC[rowInc+j+m_numElem-edgeElems*edgeElems] |=
          ZETA_P_FREE;
      }
      else {
        m_elemBC[rowInc+j+m_numElem-edgeElems*edgeElems] |=
          ZETA_P_COMM ;
        m_lzetap[rowInc+j+m_numElem-edgeElems*edgeElems] =
          ghostIdx[1] + rowInc + j ;
      }

      if (m_rowLoc == 0) {
        m_elemBC[planeInc+j] |= ETA_M_SYMM ;
      }
      else {
        m_elemBC[planeInc+j] |= ETA_M_COMM ;
        m_letam[planeInc+j] = ghostIdx[2] + rowInc + j ;
      }

      if (m_rowLoc == m_tp-1) {
        m_elemBC[planeInc+j+edgeElems*edgeElems-edgeElems] |= 
          ETA_P_FREE ;
      }
      else {
        m_elemBC[planeInc+j+edgeElems*edgeElems-edgeElems] |= 
          ETA_P_COMM ;
        m_letap[planeInc+j+edgeElems*edgeElems-edgeElems] =
          ghostIdx[3] +  rowInc + j ;
      }

      if (m_colLoc == 0) {
        m_elemBC[planeInc+j*edgeElems] |= XI_M_SYMM ;
      }
      else {
        m_elemBC[planeInc+j*edgeElems] |= XI_M_COMM ;
        m_lxim[planeInc+j*edgeElems] = ghostIdx[4] + rowInc + j ;
      }

      if (m_colLoc == m_tp-1) {
        m_elemBC[planeInc+j*edgeElems+edgeElems-1] |= XI_P_FREE ;
      }
      else {
        m_elemBC[planeInc+j*edgeElems+edgeElems-1] |= XI_P_COMM ;
        m_lxip[planeInc+j*edgeElems+edgeElems-1] =
          ghostIdx[5] + rowInc + j ;
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////
void
Domain::AllocateGradients(Int_t numElem)
{
  // Velocity gradients
  m_delv_xi.resize(numElem) ;
  m_delv_eta.resize(numElem);
  m_delv_zeta.resize(numElem) ;

  // Position gradients
  m_delx_xi.resize(numElem) ;
  m_delx_eta.resize(numElem) ;
  m_delx_zeta.resize(numElem) ;
}

///////////////////////////////////////////////////////////////////////////
void
Domain::AllocateRegionTmps(Int_t numElem)
{
  m_e_old.resize(numElem) ;
  m_delvc.resize(numElem) ;
  m_p_old.resize(numElem) ;
  m_q_old.resize(numElem) ;
  m_compression.resize(numElem) ;
  m_compHalfStep.resize(numElem) ;
  m_qq_old.resize(numElem) ;
  m_ql_old.resize(numElem) ;
  m_work.resize(numElem) ;
  m_p_new.resize(numElem) ;
  m_e_new.resize(numElem) ;
  m_q_new.resize(numElem) ;
  m_bvc.resize(numElem) ;
  m_pbvc.resize(numElem) ;
}

///////////////////////////////////////////////////////////////////////////
void
Domain::DeallocateGradients()
{
  m_delx_zeta.erase(m_delx_zeta.begin(), m_delx_zeta.end()) ;
  m_delx_eta.erase(m_delx_eta.begin(), m_delx_eta.end()) ;
  m_delx_xi.erase(m_delx_xi.begin(), m_delx_xi.end()) ;

  m_delv_zeta.erase(m_delv_zeta.begin(), m_delv_zeta.end()) ;
  m_delv_eta.erase(m_delv_eta.begin(), m_delv_eta.end()) ;
  m_delv_xi.erase(m_delv_xi.begin(), m_delv_xi.end()) ;
}

///////////////////////////////////////////////////////////////////////////
void
Domain::AllocateStrains(Int_t numElem)
{
  // principal strains
  m_dxx.resize(numElem) ; 
  m_dyy.resize(numElem) ;
  m_dzz.resize(numElem) ;
}

///////////////////////////////////////////////////////////////////////////
void
Domain::DeallocateStrains()
{
  m_dzz.erase(m_dzz.begin(), m_dzz.end()) ;
  m_dyy.erase(m_dyy.begin(), m_dyy.end()) ;
  m_dxx.erase(m_dxx.begin(), m_dxx.end()) ;
}


///////////////////////////////////////////////////////////////////////////
void InitMeshDecomp(int numRanks, int myRank,
                    int& col, int& row, int& plane, int &side)
{
  int testProcs;
  int dx, dy, dz;
  int myDom;

  // Assume cube processor layout for now 
  testProcs = (int) (cbrt(Real_t(numRanks))+0.5) ;
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
  int remainder = dx*dy*dz % numRanks ;
  if (myRank < remainder) {
    myDom = myRank*( 1+ (dx*dy*dz / numRanks)) ;
  }
  else {
    myDom = remainder*( 1+ (dx*dy*dz / numRanks)) +
      (myRank - remainder)*(dx*dy*dz/numRanks) ;
  }

  col = myDom % dx ;
  row = (myDom / dx) % dy ;
  plane = myDom / (dx*dy) ;
  side = testProcs;

  return;
}

void
Domain::ReleaseDeviceMem()
{
#ifdef _OPENACC
  acc_shutdown(acc_device_nvidia);
#endif
}

