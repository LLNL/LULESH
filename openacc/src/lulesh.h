#if !defined(USE_MPI)
# error "You should specify USE_MPI=0 or USE_MPI=1 on the compile line"
#endif


// OpenMP will be compiled in if this flag is set to 1 AND the compiler beging
// used supports it (i.e. the _OPENMP symbol is defined)
#define USE_OMP 1

#if USE_MPI
#include <mpi.h>

/*
  define one of these three symbols:

  SEDOV_SYNC_POS_VEL_NONE
  SEDOV_SYNC_POS_VEL_EARLY
  SEDOV_SYNC_POS_VEL_LATE
*/

#define SEDOV_SYNC_POS_VEL_LATE 1
#endif // if USE_MPI

#ifdef _OPENACC
#include "openacc.h"
#endif

#include <math.h>
#include <vector>

//**************************************************
// Allow flexibility for arithmetic representations 
//**************************************************

#define MAX(a, b) ( ((a) > (b)) ? (a) : (b))


// Could also support fixed point and interval arithmetic types
typedef float        real4 ;
typedef double       real8 ;
typedef long double  real10 ;  // 10 bytes on x86

typedef int    Index_t ; // array subscript and loop index
typedef real8  Real_t ;  // floating point representation
typedef int    Int_t ;   // integer representation

enum { VolumeError = -1, QStopError = -2 } ;

inline real4  SQRT(real4  arg) { return sqrtf(arg) ; }
inline real8  SQRT(real8  arg) { return sqrt(arg) ; }
inline real10 SQRT(real10 arg) { return sqrtl(arg) ; }

inline real4  CBRT(real4  arg) { return cbrtf(arg) ; }
inline real8  CBRT(real8  arg) { return cbrt(arg) ; }
inline real10 CBRT(real10 arg) { return cbrtl(arg) ; }

inline real4  FABS(real4  arg) { return fabsf(arg) ; }
inline real8  FABS(real8  arg) { return fabs(arg) ; }
inline real10 FABS(real10 arg) { return fabsl(arg) ; }

// Stuff needed for boundary conditions
// 2 BCs on each of 6 hexahedral faces (12 bits)
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

// MPI Message Tags
#define MSG_COMM_SBN      1024
#define MSG_SYNC_POS_VEL  2048
#define MSG_MONOQ         3072

#define MAX_FIELDS_PER_MPI_COMM 6

// Assume 128 byte coherence
// Assume Real_t is an "integral power of 2" bytes wide
#define CACHE_COHERENCE_PAD_REAL (128 / sizeof(Real_t))

#define CACHE_ALIGN_REAL(n) \
  (((n) + (CACHE_COHERENCE_PAD_REAL - 1)) & ~(CACHE_COHERENCE_PAD_REAL-1))

//////////////////////////////////////////////////////
// Primary data structure
//////////////////////////////////////////////////////

class Domain {

  public:

  // Constructor
  Domain(Int_t numRanks, Index_t colLoc,
      Index_t rowLoc, Index_t planeLoc,
      Index_t nx, int tp, int nr, int balance, int cost);

  void AllocateRegionTmps(Int_t numElem);
  void AllocateGradients(Int_t numElem);
  void DeallocateGradients();
  void AllocateStrains(Int_t numElem);
  void DeallocateStrains();

  // Node-centered

  // Nodal coordinates
  Real_t& x(Index_t idx)    { return m_x[idx] ; }
  Real_t& y(Index_t idx)    { return m_y[idx] ; }
  Real_t& z(Index_t idx)    { return m_z[idx] ; }
  Real_t* x()               { return &m_x[0] ; }
  Real_t* y()               { return &m_y[0] ; }
  Real_t* z()               { return &m_z[0] ; }

  // Nodal velocities
  Real_t& xd(Index_t idx)   { return m_xd[idx] ; }
  Real_t& yd(Index_t idx)   { return m_yd[idx] ; }
  Real_t& zd(Index_t idx)   { return m_zd[idx] ; }
  Real_t* xd()              { return &m_xd[0] ; }
  Real_t* yd()              { return &m_yd[0] ; }
  Real_t* zd()              { return &m_zd[0] ; }

  // Nodal accelerations
  Real_t& xdd(Index_t idx)  { return m_xdd[idx] ; }
  Real_t& ydd(Index_t idx)  { return m_ydd[idx] ; }
  Real_t& zdd(Index_t idx)  { return m_zdd[idx] ; }
  Real_t* xdd()             { return &m_xdd[0] ; }
  Real_t* ydd()             { return &m_ydd[0] ; }
  Real_t* zdd()             { return &m_zdd[0] ; }

  // Nodal forces
  Real_t& fx(Index_t idx)   { return m_fx[idx] ; }
  Real_t& fy(Index_t idx)   { return m_fy[idx] ; }
  Real_t& fz(Index_t idx)   { return m_fz[idx] ; }
  Real_t* fx()              { return &m_fx[0] ; }
  Real_t* fy()              { return &m_fy[0] ; }
  Real_t* fz()              { return &m_fz[0] ; }

  // Temporaries -- these are normally tmp arrays, but allocated globally to
  // optimize OpenACC versions of code
  Real_t* fx_elem()         { return &m_fx_elem[0] ; }
  Real_t* fy_elem()         { return &m_fy_elem[0] ; }
  Real_t* fz_elem()         { return &m_fz_elem[0] ; }
  Real_t* dvdx()            { return &m_dvdx[0] ; }
  Real_t* dvdy()            { return &m_dvdy[0] ; }
  Real_t* dvdz()            { return &m_dvdz[0] ; }
  Real_t* x8n()             { return &m_x8n[0] ; }
  Real_t* y8n()             { return &m_y8n[0] ; }
  Real_t* z8n()             { return &m_z8n[0] ; }
  Real_t* sigxx()           { return &m_sigxx[0] ; }
  Real_t* sigyy()           { return &m_sigyy[0] ; }
  Real_t* sigzz()           { return &m_sigzz[0] ; }
  Real_t* determ()          { return &m_determ[0] ; }
  Real_t* vnew()            { return &m_vnew[0] ; }
  Real_t* e_old()           { return &m_e_old[0] ; }
  Real_t* delvc()           { return &m_delvc[0] ; }
  Real_t* p_old()           { return &m_p_old[0] ; }
  Real_t* q_old()           { return &m_q_old[0] ; }
  Real_t* compression()     { return &m_compression[0] ; }
  Real_t* compHalfStep()    { return &m_compHalfStep[0] ; }
  Real_t* qq_old()          { return &m_qq_old[0] ; }
  Real_t* ql_old()          { return &m_ql_old[0] ; }
  Real_t* work()            { return &m_work[0] ; }
  Real_t* p_new()           { return &m_p_new[0] ; }
  Real_t* e_new()           { return &m_e_new[0] ; }
  Real_t* q_new()           { return &m_q_new[0] ; }
  Real_t* bvc()             { return &m_bvc[0] ; }
  Real_t* pbvc()            { return &m_pbvc[0] ; }

  // Nodal mass
  Real_t& nodalMass(Index_t idx) { return m_nodalMass[idx] ; }
  Real_t* nodalMass()            { return &m_nodalMass[0] ; }

  // Nodes on symmertry planes
  Index_t symmX(Index_t idx) { return m_symmX[idx] ; }
  Index_t symmY(Index_t idx) { return m_symmY[idx] ; }
  Index_t symmZ(Index_t idx) { return m_symmZ[idx] ; }
  Index_t* symmX()           { return &m_symmX[0] ; }
  Index_t* symmY()           { return &m_symmY[0] ; }
  Index_t* symmZ()           { return &m_symmZ[0] ; }
  bool symmXempty()          { return m_symmX.empty(); }
  bool symmYempty()          { return m_symmY.empty(); }
  bool symmZempty()          { return m_symmZ.empty(); }

  //
  // Element-centered
  //
  Index_t&  regElemSize(Index_t idx) { return m_regElemSize[idx] ; }
  Index_t&  regNumList(Index_t idx) { return m_regNumList[idx] ; }
  Index_t*  regNumList()            { return &m_regNumList[0] ; }
  Index_t*  regElemlist(Int_t r)    { return m_regElemlist[r] ; }
  Index_t&  regElemlist(Int_t r, Index_t idx) { return m_regElemlist[r][idx] ; }

  Index_t*  nodelist(Index_t idx)    { return &m_nodelist[Index_t(8)*idx] ; }
  Index_t*  nodelist()               { return &m_nodelist[0] ; }

  // elem connectivities through face
  Index_t&  lxim(Index_t idx) { return m_lxim[idx] ; }
  Index_t*  lxim()            { return &m_lxim[0] ; }
  Index_t&  lxip(Index_t idx) { return m_lxip[idx] ; }
  Index_t*  lxip()            { return &m_lxip[0] ; }
  Index_t&  letam(Index_t idx) { return m_letam[idx] ; }
  Index_t*  letam()            { return &m_letam[0] ; }
  Index_t&  letap(Index_t idx) { return m_letap[idx] ; }
  Index_t*  letap()            { return &m_letap[0] ; }
  Index_t&  lzetam(Index_t idx) { return m_lzetam[idx] ; }
  Index_t*  lzetam()            { return &m_lzetam[0] ; }
  Index_t&  lzetap(Index_t idx) { return m_lzetap[idx] ; }
  Index_t*  lzetap()            { return &m_lzetap[0] ; }

  // elem face symm/free-surface flag
  Int_t&  elemBC(Index_t idx) { return m_elemBC[idx] ; }
  Int_t*  elemBC()            { return &m_elemBC[0] ; }

  // Principal strains - temporary
  Real_t& dxx(Index_t idx)  { return m_dxx[idx] ; }
  Real_t& dyy(Index_t idx)  { return m_dyy[idx] ; }
  Real_t& dzz(Index_t idx)  { return m_dzz[idx] ; }
  Real_t* dxx()             { return &m_dxx[0] ; }
  Real_t* dyy()             { return &m_dyy[0] ; }
  Real_t* dzz()             { return &m_dzz[0] ; }

  // Velocity gradient - temporary
  Real_t& delv_xi(Index_t idx)    { return m_delv_xi[idx] ; }
  Real_t& delv_eta(Index_t idx)   { return m_delv_eta[idx] ; }
  Real_t& delv_zeta(Index_t idx)  { return m_delv_zeta[idx] ; }
  Real_t* delv_xi()               { return &m_delv_xi[0] ; }
  Real_t* delv_eta()              { return &m_delv_eta[0] ; }
  Real_t* delv_zeta()             { return &m_delv_zeta[0] ; }

  // Position gradient - temporary
  Real_t& delx_xi(Index_t idx)    { return m_delx_xi[idx] ; }
  Real_t& delx_eta(Index_t idx)   { return m_delx_eta[idx] ; }
  Real_t& delx_zeta(Index_t idx)  { return m_delx_zeta[idx] ; }
  Real_t* delx_xi()               { return &m_delx_xi[0] ; }
  Real_t* delx_eta()              { return &m_delx_eta[0] ; }
  Real_t* delx_zeta()             { return &m_delx_zeta[0] ; }

  // Energy
  Real_t& e(Index_t idx)          { return m_e[idx] ; }
  Real_t* e()                     { return &m_e[0] ; }

  // Pressure
  Real_t& p(Index_t idx)          { return m_p[idx] ; }
  Real_t* p()                     { return &m_p[0] ; }

  // Artificial viscosity
  Real_t& q(Index_t idx)          { return m_q[idx] ; }
  Real_t* q()                     { return &m_q[0] ; }

  // Linear term for q
  Real_t& ql(Index_t idx)         { return m_ql[idx] ; }
  Real_t* ql()                    { return &m_ql[0] ; }
  // Quadratic term for q
  Real_t& qq(Index_t idx)         { return m_qq[idx] ; }
  Real_t* qq()                    { return &m_qq[0] ; }

  // Relative volume
  Real_t& v(Index_t idx)          { return m_v[idx] ; }
  Real_t* v()                     { return &m_v[0] ; }
  Real_t& delv(Index_t idx)       { return m_delv[idx] ; }
  Real_t* delv()                  { return &m_delv[0] ; }

  // Reference volume
  Real_t& volo(Index_t idx)       { return m_volo[idx] ; }
  Real_t* volo()                  { return &m_volo[0] ; }

  // volume derivative over volume
  Real_t& vdov(Index_t idx)       { return m_vdov[idx] ; }
  Real_t* vdov()                  { return &m_vdov[0] ; }

  // Element characteristic length
  Real_t& arealg(Index_t idx)     { return m_arealg[idx] ; }
  Real_t* arealg()                  { return &m_arealg[0] ; }

  // Sound speed
  Real_t& ss(Index_t idx)         { return m_ss[idx] ; }
  Real_t* ss()                    { return &m_ss[0] ; }

  // Element mass
  Real_t& elemMass(Index_t idx)  { return m_elemMass[idx] ; }
  Real_t* elemMass()             { return &m_elemMass[0] ; }


#if USE_MPI   
  // Communication Work space 
  Real_t *commDataSend ;
  Real_t *commDataRecv ;

  // Maximum number of block neighbors 
  MPI_Request recvRequest[26] ; // 6 faces + 12 edges + 8 corners 
  MPI_Request sendRequest[26] ; // 6 faces + 12 edges + 8 corners 
#endif

  // Parameters 

  // Cutoffs
  Real_t u_cut() const               { return m_u_cut ; }
  Real_t e_cut() const               { return m_e_cut ; }
  Real_t p_cut() const               { return m_p_cut ; }
  Real_t q_cut() const               { return m_q_cut ; }
  Real_t v_cut() const               { return m_v_cut ; }

  // Other constants (usually are settable via input file in real codes)
  Real_t hgcoef() const              { return m_hgcoef ; }
  Real_t qstop() const               { return m_qstop ; }
  Real_t monoq_max_slope() const     { return m_monoq_max_slope ; }
  Real_t monoq_limiter_mult() const  { return m_monoq_limiter_mult ; }
  Real_t ss4o3() const               { return m_ss4o3 ; }
  Real_t qlc_monoq() const           { return m_qlc_monoq ; }
  Real_t qqc_monoq() const           { return m_qqc_monoq ; }
  Real_t qqc() const                 { return m_qqc ; }

  Real_t eosvmax() const             { return m_eosvmax ; }
  Real_t eosvmin() const             { return m_eosvmin ; }
  Real_t pmin() const                { return m_pmin ; }
  Real_t emin() const                { return m_emin ; }
  Real_t dvovmax() const             { return m_dvovmax ; }
  Real_t refdens() const             { return m_refdens ; }

  Index_t *nodeElemCount() const      { return m_nodeElemCount ; }
  Index_t *nodeElemStart() const      { return m_nodeElemStart ; }
  Index_t *nodeElemCornerList() const { return m_nodeElemCornerList ; }

  // Timestep controls, etc...
  Real_t& time()                 { return m_time ; }
  Real_t& deltatime()            { return m_deltatime ; }
  Real_t& deltatimemultlb()      { return m_deltatimemultlb ; }
  Real_t& deltatimemultub()      { return m_deltatimemultub ; }
  Real_t& stoptime()             { return m_stoptime ; }
  Real_t& dtcourant()            { return m_dtcourant ; }
  Real_t& dthydro()              { return m_dthydro ; }
  Real_t& dtmax()                { return m_dtmax ; }
  Real_t& dtfixed()              { return m_dtfixed ; }

  Int_t&  cycle()                { return m_cycle ; }
  Index_t&  numRanks()           { return m_numRanks ; }

  Index_t&  colLoc()             { return m_colLoc ; }
  Index_t&  rowLoc()             { return m_rowLoc ; }
  Index_t&  planeLoc()           { return m_planeLoc ; }
  Index_t&  tp()                 { return m_tp ; }

  Index_t&  sizeX()              { return m_sizeX ; }
  Index_t&  sizeY()              { return m_sizeY ; }
  Index_t&  sizeZ()              { return m_sizeZ ; }
  Index_t&  numReg()             { return m_numReg ; }
  Int_t&  cost()             { return m_cost ; }
  Index_t&  numElem()            { return m_numElem ; }
  Index_t&  numNode()            { return m_numNode ; }

  Index_t&  maxPlaneSize()       { return m_maxPlaneSize ; }
  Index_t&  maxEdgeSize()        { return m_maxEdgeSize ; }

  // OpenACC
  void ReleaseDeviceMem();
  int m_numDevs;

  private:

  void BuildMesh(Int_t nx, Int_t edgeNodes, Int_t edgeElems);
  void SetupThreadSupportStructures();
  void CreateRegionIndexSets(Int_t nreg, Int_t balance);
  void SetupCommBuffers(Int_t edgeNodes);
  void SetupSymmetryPlanes(Int_t edgeNodes);
  void SetupElementConnectivities(Int_t edgeElems);
  void SetupBoundaryConditions(Int_t edgeElems);

  /* Node-centered */
  std::vector<Real_t> m_x ;  /* coordinates */
  std::vector<Real_t> m_y ;
  std::vector<Real_t> m_z ;

  std::vector<Real_t> m_xd ; /* velocities */
  std::vector<Real_t> m_yd ;
  std::vector<Real_t> m_zd ;

  std::vector<Real_t> m_xdd ; /* accelerations */
  std::vector<Real_t> m_ydd ;
  std::vector<Real_t> m_zdd ;

  std::vector<Real_t> m_fx ;  /* forces */
  std::vector<Real_t> m_fy ;
  std::vector<Real_t> m_fz ;

  /* tmp arrays that are allocated globally for OpenACC */
  std::vector<Real_t> m_fx_elem ;
  std::vector<Real_t> m_fy_elem ;
  std::vector<Real_t> m_fz_elem ;
  std::vector<Real_t> m_dvdx ;
  std::vector<Real_t> m_dvdy ;
  std::vector<Real_t> m_dvdz ;
  std::vector<Real_t> m_x8n ;
  std::vector<Real_t> m_y8n ;
  std::vector<Real_t> m_z8n ;
  std::vector<Real_t> m_sigxx ;
  std::vector<Real_t> m_sigyy ;
  std::vector<Real_t> m_sigzz ;
  std::vector<Real_t> m_determ ;
  std::vector<Real_t> m_e_old ;
  std::vector<Real_t> m_delvc ;
  std::vector<Real_t> m_p_old ;
  std::vector<Real_t> m_q_old ;
  std::vector<Real_t> m_compression ;
  std::vector<Real_t> m_compHalfStep ;
  std::vector<Real_t> m_qq_old ;
  std::vector<Real_t> m_ql_old ;
  std::vector<Real_t> m_work ;
  std::vector<Real_t> m_p_new ;
  std::vector<Real_t> m_e_new ;
  std::vector<Real_t> m_q_new ;
  std::vector<Real_t> m_bvc ;
  std::vector<Real_t> m_pbvc ;

  std::vector<Real_t> m_nodalMass ;  /* mass */

  std::vector<Index_t> m_symmX ;  /* symmetry plane nodesets */
  std::vector<Index_t> m_symmY ;
  std::vector<Index_t> m_symmZ ;

  // Element-centered

  // Region information
  Int_t    m_numReg ;
  Int_t    m_cost; //imbalance cost
  Int_t   *m_regElemSize ;   // Size of region sets
  Index_t *m_regNumList ;    // Region number per domain element
  Index_t **m_regElemlist ;  // region indexset 

  std::vector<Index_t>  m_matElemlist ;  /* material indexset */
  std::vector<Index_t>  m_nodelist ;     /* elemToNode connectivity */

  std::vector<Index_t>  m_lxim ;  /* element connectivity across each face */
  std::vector<Index_t>  m_lxip ;
  std::vector<Index_t>  m_letam ;
  std::vector<Index_t>  m_letap ;
  std::vector<Index_t>  m_lzetam ;
  std::vector<Index_t>  m_lzetap ;

  std::vector<Int_t>    m_elemBC ;  /* symmetry/free-surface flags for each elem face */

  std::vector<Real_t> m_dxx ;  /* principal strains -- temporary */
  std::vector<Real_t> m_dyy ;
  std::vector<Real_t> m_dzz ;

  std::vector<Real_t> m_delv_xi ;    /* velocity gradient -- temporary */
  std::vector<Real_t> m_delv_eta ;
  std::vector<Real_t> m_delv_zeta ;

  std::vector<Real_t> m_delx_xi ;    /* coordinate gradient -- temporary */
  std::vector<Real_t> m_delx_eta ;
  std::vector<Real_t> m_delx_zeta ;

  std::vector<Real_t> m_e ;   /* energy */

  std::vector<Real_t> m_p ;   /* pressure */
  std::vector<Real_t> m_q ;   /* q */
  std::vector<Real_t> m_ql ;  /* linear term for q */
  std::vector<Real_t> m_qq ;  /* quadratic term for q */

  std::vector<Real_t> m_v ;     /* relative volume */
  std::vector<Real_t> m_volo ;  /* reference volume */
  std::vector<Real_t> m_vnew ;  /* new relative volume -- temporary */
  std::vector<Real_t> m_delv ;  /* m_vnew - m_v */
  std::vector<Real_t> m_vdov ;  /* volume derivative over volume */

  std::vector<Real_t> m_arealg ;  /* characteristic length of an element */

  std::vector<Real_t> m_ss ;      /* "sound speed" */

  std::vector<Real_t> m_elemMass ;  /* mass */

  // Cutoffs (treat as constants)
  const Real_t  m_e_cut ;             // energy tolerance 
  const Real_t  m_p_cut ;             // pressure tolerance 
  const Real_t  m_q_cut ;             // q tolerance 
  const Real_t  m_v_cut ;             // relative volume tolerance 
  const Real_t  m_u_cut ;             // velocity tolerance 

  // Other constants (usually setable, but hardcoded in this proxy app)

  const Real_t  m_hgcoef ;            // hourglass control 
  const Real_t  m_ss4o3 ;
  const Real_t  m_qstop ;             // excessive q indicator 
  const Real_t  m_monoq_max_slope ;
  const Real_t  m_monoq_limiter_mult ;
  const Real_t  m_qlc_monoq ;         // linear term coef for q 
  const Real_t  m_qqc_monoq ;         // quadratic term coef for q 
  const Real_t  m_qqc ;
  const Real_t  m_eosvmax ;
  const Real_t  m_eosvmin ;
  const Real_t  m_pmin ;              // pressure floor 
  const Real_t  m_emin ;              // energy floor 
  const Real_t  m_dvovmax ;           // maximum allowable volume change 
  const Real_t  m_refdens ;           // reference density 

  // Variables to keep track of timestep, simulation time, and cycle
  Real_t  m_dtcourant ;         // courant constraint 
  Real_t  m_dthydro ;           // volume change constraint 
  Int_t   m_cycle ;             // iteration count for simulation 
  Real_t  m_dtfixed ;           // fixed time increment 
  Real_t  m_time ;              // current time 
  Real_t  m_deltatime ;         // variable time increment 
  Real_t  m_deltatimemultlb ;
  Real_t  m_deltatimemultub ;
  Real_t  m_dtmax ;             // maximum allowable time increment 
  Real_t  m_stoptime ;          // end time for simulation 


  Int_t   m_numRanks ;


  Index_t m_colLoc ;
  Index_t m_rowLoc ;
  Index_t m_planeLoc ;
  Index_t m_tp ;

  Index_t m_sizeX ;
  Index_t m_sizeY ;
  Index_t m_sizeZ ;
  Index_t m_numElem ;
  Index_t m_numNode ;

  Index_t m_maxPlaneSize ;
  Index_t m_maxEdgeSize ;

  // OMP hack 
  Index_t *m_nodeElemCount ;
  Index_t *m_nodeElemStart ;
  Index_t *m_nodeElemCornerList ;

  // Used in setup
  Index_t m_rowMin, m_rowMax;
  Index_t m_colMin, m_colMax;
  Index_t m_planeMin, m_planeMax ;

} ;

struct cmdLineOpts {
  int its; // -i 
  int nx;  // -s 
  int numReg; // -r 
  int numFiles; // -f
  int showProg; // -p
  int quiet; // -q
  int viz; // -v 
  Int_t cost; // -c
  int balance; // -b
};



// Function Prototypes

// lulesh-par
Real_t CalcElemVolume( const Real_t x[8],
    const Real_t y[8],
    const Real_t z[8]);

// lulesh-util
void ParseCommandLineOptions(int argc, char *argv[],
    int myRank, struct cmdLineOpts *opts);
void VerifyAndWriteFinalOutput(Real_t elapsed_time,
    Domain& locDom,
    Int_t nx,
    Int_t numRanks);

// lulesh-viz
//void DumpToVisit(Domain& domain, int numFiles, int myRank, int numRanks);

// lulesh-comm
void CommRecv(Domain& domain, int msgType, Index_t xferFields,
    Index_t dx, Index_t dy, Index_t dz,
    bool doRecv, bool planeOnly);
void CommSend(Domain& domain, int msgType,
    Index_t xferFields, Real_t **fieldData,
    Index_t dx, Index_t dy, Index_t dz,
    bool doSend, bool planeOnly);
void CommSBN(Domain& domain, int xferFields, Real_t **fieldData);
void CommSyncPosVel(Domain& domain);
void CommMonoQ(Domain& domain);

// lulesh-init
void InitMeshDecomp(int numRanks, int myRank,
    int &col, int& row, int& plane, int& side);

