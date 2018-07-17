#include "vector.h"

#define LULESH_SHOW_PROGRESS 0
#define DOUBLE_PRECISION
//#define SAMI 

#if USE_MPI
#include <mpi.h>

/*
   define one of these three symbols:

   SEDOV_SYNC_POS_VEL_NONE
   SEDOV_SYNC_POS_VEL_EARLY
   SEDOV_SYNC_POS_VEL_LATE
*/

// TODO: currently we support only early sync!
#define SEDOV_SYNC_POS_VEL_EARLY 1
#endif

enum {
  VolumeError = -1,
  QStopError = -2,
  LFileError = -3
} ;

/* Could also support fixed point and interval arithmetic types */
typedef float        real4 ;
typedef double       real8 ;

typedef int    Index_t ; /* array subscript and loop index */
typedef int    Int_t ;   /* integer representation */
#ifdef DOUBLE_PRECISION
typedef real8  Real_t ;  /* floating point representation */
#else
typedef real4  Real_t ;  /* floating point representation */
#endif

class Domain
{

public: 

  void sortRegions(Vector_h<Int_t>& regReps_h, Vector_h<Index_t>& regSorted_h);
  void CreateRegionIndexSets(Int_t nr, Int_t balance);


  Index_t max_streams;
  std::vector<cudaStream_t> streams;

  /* Elem-centered */

  Vector_d<Index_t> matElemlist ; /* material indexset */
  Vector_d<Index_t> nodelist ;    /* elemToNode connectivity */

  Vector_d<Index_t> lxim ;        /* element connectivity through face */
  Vector_d<Index_t> lxip ;
  Vector_d<Index_t> letam ;
  Vector_d<Index_t> letap ;
  Vector_d<Index_t> lzetam ;
  Vector_d<Index_t> lzetap ;

  Vector_d<Int_t> elemBC ;        /* elem face symm/free-surf flag */

  Vector_d<Real_t> e ;            /* energy */

  Vector_d<Real_t> p ;            /* pressure */

  Vector_d<Real_t> q ;            /* q */
  Vector_d<Real_t> ql ;           /* linear term for q */
  Vector_d<Real_t> qq ;           /* quadratic term for q */

  Vector_d<Real_t> v ;            /* relative volume */

  Vector_d<Real_t> volo ;         /* reference volume */
  Vector_d<Real_t> delv ;         /* m_vnew - m_v */
  Vector_d<Real_t> vdov ;         /* volume derivative over volume */

  Vector_d<Real_t> arealg ;       /* char length of an element */
  
  Vector_d<Real_t> ss ;           /* "sound speed" */

  Vector_d<Real_t> elemMass ;     /* mass */

  Vector_d<Real_t>* vnew ;         /* new relative volume -- temporary */

  Vector_d<Real_t>* delv_xi ;      /* velocity gradient -- temporary */
  Vector_d<Real_t>* delv_eta ;
  Vector_d<Real_t>* delv_zeta ;

  Vector_d<Real_t>* delx_xi ;      /* coordinate gradient -- temporary */
  Vector_d<Real_t>* delx_eta ;
  Vector_d<Real_t>* delx_zeta ;

  Vector_d<Real_t>* dxx ;          /* principal strains -- temporary */
  Vector_d<Real_t>* dyy ;
  Vector_d<Real_t>* dzz ;

  /* Node-centered */

  Vector_d<Real_t> x ;            /* coordinates */
  Vector_d<Real_t> y ;
  Vector_d<Real_t> z ;

  Vector_d<Real_t> xd ;           /* velocities */
  Vector_d<Real_t> yd ;
  Vector_d<Real_t> zd ;


  Vector_d<Real_t> xdd ;          /* accelerations */
  Vector_d<Real_t> ydd ;
  Vector_d<Real_t> zdd ;

  Vector_d<Real_t> fx ;           /* forces */
  Vector_d<Real_t> fy ;
  Vector_d<Real_t> fz ;

  Vector_d<Real_t> nodalMass ;    /* mass */
  Vector_h<Real_t> h_nodalMass ;    /* mass - host */

  /* device pointers for comms */
  Real_t *d_delv_xi ;      /* velocity gradient -- temporary */
  Real_t *d_delv_eta ;
  Real_t *d_delv_zeta ;

  Real_t *d_x ;            /* coordinates */
  Real_t *d_y ;
  Real_t *d_z ;

  Real_t *d_xd ;           /* velocities */
  Real_t *d_yd ;
  Real_t *d_zd ;

  Real_t *d_fx ;           /* forces */
  Real_t *d_fy ;
  Real_t *d_fz ;

  /* access elements for comms */
  Real_t& get_delv_xi(Index_t idx) { return d_delv_xi[idx] ; }
  Real_t& get_delv_eta(Index_t idx) { return d_delv_eta[idx] ; }
  Real_t& get_delv_zeta(Index_t idx) { return d_delv_zeta[idx] ; }

  Real_t& get_x(Index_t idx) { return d_x[idx] ; }
  Real_t& get_y(Index_t idx) { return d_y[idx] ; }
  Real_t& get_z(Index_t idx) { return d_z[idx] ; }

  Real_t& get_xd(Index_t idx) { return d_xd[idx] ; }
  Real_t& get_yd(Index_t idx) { return d_yd[idx] ; }
  Real_t& get_zd(Index_t idx) { return d_zd[idx] ; }

  Real_t& get_fx(Index_t idx) { return d_fx[idx] ; }
  Real_t& get_fy(Index_t idx) { return d_fy[idx] ; }
  Real_t& get_fz(Index_t idx) { return d_fz[idx] ; }

  // host access
  Real_t& get_nodalMass(Index_t idx) { return h_nodalMass[idx] ; }

  /* Boundary nodesets */

  Vector_d<Index_t> symmX ;       /* symmetry plane nodesets */
  Vector_d<Index_t> symmY ;        
  Vector_d<Index_t> symmZ ;
   
  Vector_d<Int_t> nodeElemCount ;
  Vector_d<Int_t> nodeElemStart;
  Vector_d<Index_t> nodeElemCornerList ;

  /* Parameters */

  Real_t dtfixed ;               /* fixed time increment */
  Real_t deltatimemultlb ;
  Real_t deltatimemultub ;
  Real_t stoptime ;              /* end time for simulation */
  Real_t dtmax ;                 /* maximum allowable time increment */
  Int_t cycle ;                  /* iteration count for simulation */

  Real_t* dthydro_h;             /* hydro time constraint */ 
  Real_t* dtcourant_h;           /* courant time constraint */
  Index_t* bad_q_h;              /* flag to indicate Q error */
  Index_t* bad_vol_h;            /* flag to indicate volume error */

  /* cuda Events to indicate completion of certain kernels */
  cudaEvent_t time_constraint_computed;

  Real_t time_h ;               /* current time */
  Real_t deltatime_h ;          /* variable time increment */

  Real_t u_cut ;                /* velocity tolerance */
  Real_t hgcoef ;               /* hourglass control */
  Real_t qstop ;                /* excessive q indicator */
  Real_t monoq_max_slope ;
  Real_t monoq_limiter_mult ;   
  Real_t e_cut ;                /* energy tolerance */
  Real_t p_cut ;                /* pressure tolerance */
  Real_t ss4o3 ;
  Real_t q_cut ;                /* q tolerance */
  Real_t v_cut ;                /* relative volume tolerance */
  Real_t qlc_monoq ;            /* linear term coef for q */
  Real_t qqc_monoq ;            /* quadratic term coef for q */
  Real_t qqc ;
  Real_t eosvmax ;
  Real_t eosvmin ;
  Real_t pmin ;                 /* pressure floor */
  Real_t emin ;                 /* energy floor */
  Real_t dvovmax ;              /* maximum allowable volume change */
  Real_t refdens ;              /* reference density */

   Index_t m_colLoc ;
   Index_t m_rowLoc ;
   Index_t m_planeLoc ;
   Index_t m_tp ;

   Index_t&  colLoc()             { return m_colLoc ; }
   Index_t&  rowLoc()             { return m_rowLoc ; }
   Index_t&  planeLoc()           { return m_planeLoc ; }
   Index_t&  tp()                 { return m_tp ; }

  Index_t sizeX ;
  Index_t sizeY ;
  Index_t sizeZ ;
  Index_t maxPlaneSize ;
  Index_t maxEdgeSize ;

  Index_t numElem ;
  Index_t padded_numElem ; 

  Index_t numNode;
  Index_t padded_numNode ; 

  Index_t numSymmX ; 
  Index_t numSymmY ; 
  Index_t numSymmZ ; 

  Index_t octantCorner;

   // Region information
   Int_t numReg ; //number of regions (def:11)
   Int_t balance; //Load balance between regions of a domain (def: 1)
   Int_t  cost;  //imbalance cost (def: 1)
   Int_t*   regElemSize ;   // Size of region sets
   Vector_d<Int_t> regCSR;  // records the begining and end of each region
   Vector_d<Int_t> regReps; // records the rep number per region
   Vector_d<Index_t> regNumList;    // Region number per domain element
   Vector_d<Index_t> regElemlist;  // region indexset 
   Vector_d<Index_t> regSorted; // keeps index of sorted regions
   
   //
   // MPI-Related additional data
   //

   Index_t m_numRanks;
   Index_t& numRanks() { return m_numRanks ; }

   void SetupCommBuffers(Int_t edgeNodes);
   void BuildMesh(Int_t nx, Int_t edgeNodes, Int_t edgeElems, Int_t domNodes, Int_t padded_domElems, Vector_h<Real_t> &x_h, Vector_h<Real_t> &y_h, Vector_h<Real_t> &z_h, Vector_h<Int_t> &nodelist_h);

   // Used in setup
   Index_t m_rowMin, m_rowMax;
   Index_t m_colMin, m_colMax;
   Index_t m_planeMin, m_planeMax ;

#if USE_MPI   
   // Communication Work space 
   Real_t *commDataSend ;
   Real_t *commDataRecv ;

   Real_t *d_commDataSend ;
   Real_t *d_commDataRecv ;

   // Maximum number of block neighbors 
   MPI_Request recvRequest[26] ; // 6 faces + 12 edges + 8 corners 
   MPI_Request sendRequest[26] ; // 6 faces + 12 edges + 8 corners 
#endif

};

typedef Real_t& (Domain::* Domain_member )(Index_t) ;

// Assume 128 byte coherence
// Assume Real_t is an "integral power of 2" bytes wide
#define CACHE_COHERENCE_PAD_REAL (128 / sizeof(Real_t))

#define CACHE_ALIGN_REAL(n) \
   (((n) + (CACHE_COHERENCE_PAD_REAL - 1)) & ~(CACHE_COHERENCE_PAD_REAL-1))

// MPI Message Tags
#define MSG_COMM_SBN      1024
#define MSG_SYNC_POS_VEL  2048
#define MSG_MONOQ         3072

#define MAX_FIELDS_PER_MPI_COMM 6

// cpu-comms
void CommRecv(Domain& domain, Int_t msgType, Index_t xferFields,
              Index_t dx, Index_t dy, Index_t dz,
              bool doRecv, bool planeOnly);
void CommSend(Domain& domain, Int_t msgType,
              Index_t xferFields, Domain_member *fieldData,
              Index_t dx, Index_t dy, Index_t dz,
              bool doSend, bool planeOnly);
void CommSBN(Domain& domain, Int_t xferFields, Domain_member *fieldData);
void CommSyncPosVel(Domain& domain);
void CommMonoQ(Domain& domain);

// gpu-comms
void CommSendGpu(Domain& domain, Int_t msgType,
              Index_t xferFields, Domain_member *fieldData,
              Index_t dx, Index_t dy, Index_t dz,
              bool doSend, bool planeOnly, cudaStream_t stream);
void CommSBNGpu(Domain& domain, Int_t xferFields, Domain_member *fieldData, cudaStream_t *streams);
void CommSyncPosVelGpu(Domain& domain, cudaStream_t *streams);
void CommMonoQGpu(Domain& domain, cudaStream_t stream);
