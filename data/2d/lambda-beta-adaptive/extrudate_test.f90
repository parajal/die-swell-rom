! Viscoelastic extrudate swell problem
! Planar flow for a Giesekus model
! DEVSS-G/SUPG
! Implicit bilinear terms of CE in momentum balance.
! second-order Gear time integration

! NOTE: If you refine the mesh you should also refine the inlet mesh by setting dx_inlet
! lin_elem can be set to true or false to choose linear or quadratic elements for the height
! function

program extrudate_swell2d_c

  use tfem_m
  use viscoelastic_elements_m
  use viscoelastic_elements_generic_m
  use hsl_ma41_m
  use io_utils_m
  use surface_advection_elements_m
  use figplot_m
  use update_mesh_nodes_bc_m
  use stokes_elements_m
  use math_defs_m

 use metis5_m

  implicit none


! constants flow problem

  integer, parameter :: &
    uintpl = 6,     & ! P2 velocities
    pintpl = 2,     & ! P1 pressures
    gintpl = 2,     & ! P1 gradients
    cintpl = 2,     & ! P1 conformation
    ointpl = 6,     & ! P2 shape of object elements
    gauss = 6,      & ! 6-point Gauss integration of triangles
    gaussb = 3,     & ! 3-point integration of boundary elements
    ncompc = 3,     & ! number of conformation tensor components
    ncompg = 4,     & ! number of gradient tensor components
    nmodes = 1,     & ! number of modes
    startm = 51,    & ! start of material model data
    ndim = 2,       & ! dimension of space
    coorsys = 0,    & ! axisymmetric coordinate system
    model = 3         ! 3: Giesekus model, 5: linear PTT

! constants surface advection

  integer, parameter :: &
    ninti_sf_adv = 5,   & ! number of Gauss points
    vintpl = 6,         & ! P2 velocities
    method = 1            ! discretization method 0: Galerkin, 1: SUPG

! definitions

  type(mesh_t) :: mesh
  type(input_probdef_t) :: input_probdef, input_probdefc, input_probdefc_projc
  type(problem_t), target :: problem, problemc, problemc_projc
  type(sysmatrix_t) :: sysmatrix, sysmatrixc, sysmatrixc_projc
  type(sysvector_t), target :: sol, soln, solm1
  type(sysvector_t) :: rhsd
  type(sample_t) :: sample_v, sample_c
  type(oldvectors_t) :: oldvectors_ve
  type(coefficients_t) :: coefficients
  type(sysvector_t), dimension(ncompc,nmodes), target :: solc, solcn, solcm1, &
        solc_projcn, solc_projcm1
  type(sysvector_t), dimension(ncompc,nmodes) :: rhsc, rhsc_projc
  type(lu_ma41_t) :: luc
  type(solver_options_ma41_t) :: solver_options_c, solver_options_u
  type(refinement_fields_t) :: refinement_fields
  type(vector_t) :: ctensor
  type(vector_t) :: pressure, velocity
  type(subscriptvec_t) :: cxx, cxy, cyy, cxx7, cxy7, cyy7

! 1D height function for surface advection
  
  type(meshgen_options_t) :: mesh_options
  type(mesh_t) :: mesh_sf_adv
  type(input_probdef_t) :: input_probdef_sf_adv
  type(problem_t) :: problem_sf_adv
  type(sysmatrix_t) :: sysmatrix_sf_adv
  type(sysvector_t), target :: sol_sf_adv, sol_sf_adv_n, sol_sf_adv_nm1
  type(sysvector_t), target :: rhsd_sf_adv, sol_sf_adv_pred, sol_sf_adv_pred_n
  type(oldvectors_t) :: oldvectors_sf_adv, oldvectors_sample_H, oldvectors_sf_adv_deriv
  type(coefficients_t) :: coefficients_sf_adv
  type(vector_t), target :: velocity_sf_adv, height_sf_adv
  type(subscript_t) :: hgt, hgt_end
  type(subscriptvec_t) :: velx_sf, vely_sf
  type(subscript_t) :: vel_in(2), cc_in(ncompc)
  type(subscript_t) :: velx7, vely7
  type(subscriptvec_t) :: subsc_in
  integer :: hintpl

! ALE mesh motion problem
  type(problem_t), target ::  problem_lapl
  type(vector_t), target :: meshvel
  type(subscript_t) :: subsh
  real(dp), allocatable, dimension(:,:) :: meshcoor_initial, xc
  real(dp), allocatable, dimension(:) :: Hhat, Hhatn ! interface location

! All parameters for inlet problem

  type(mesh_t) :: mesh_inlet
  type(input_probdef_t) :: input_probdef_inlet, input_probdefc_inlet, &
            input_probdefc_projc_inlet
  type(problem_t), target :: problem_inlet, problemc_inlet, problemc_projc_inlet
  type(sysmatrix_t) :: sysmatrix_inlet, sysmatrixc_inlet, sysmatrixc_projc_inlet
  type(sysvector_t), target :: sol_inlet, soln_inlet, solm1_inlet
  type(sysvector_t) :: rhsd_inlet
  type(oldvectors_t) :: oldvectors_ve_inlet
  type(coefficients_t) :: coefficients_inlet
  type(sysvector_t), dimension(ncompc,nmodes), target :: solc_inlet, solcm1_inlet, &
       solcn_inlet, solc_projcm1_inlet, solc_projcn_inlet
  type(sysvector_t), dimension(ncompc,nmodes) :: rhsc_inlet, rhsc_projc_inlet
  type(lu_ma41_t) :: luc_inlet
  type(solver_options_ma41_t) :: solver_options_u_in, solver_options_c_in
  real(dp), allocatable, dimension(:,:) :: vel_inlet, c_inlet

  integer :: physqgrad, physqvel, physqpress
 
! type definitions for projection of velocity gradients (if DEVSS is not used)

  type(input_probdef_t) :: input_probdef_grad
  type(problem_t) :: problem_grad
  type(sysmatrix_t) :: sysmatrix_grad
  type(sysvector_t) :: sol_grad, rhsd_grad(ncompg)
  type(oldvectors_t) :: oldvectors_grad
  type(vector_t), target :: gradients

  type(input_probdef_t) :: input_probdef_grad_inlet
  type(problem_t) :: problem_grad_inlet
  type(sysmatrix_t) :: sysmatrix_grad_inlet
  type(sysvector_t) :: sol_grad_inlet, rhsd_grad_inlet(ncompg)
  type(oldvectors_t) :: oldvectors_grad_inlet
  type(vector_t), target :: gradients_inlet

! variables

  integer :: &
    timeint1 = 1,          & ! (first-order) Euler time integration (first step)
    timeint2 = 7,          & ! (second-order) semi-implicit Gear time integration
    numtimesteps = 1000,   & ! number of time steps
    step0 = 0,             & ! initial step number
    htype = 2,             & ! upwind parameter
    Uscaling = 3,          & ! upwind parameter
    vtkevery = 100,          & ! plot vtk every .. steps
    logc = 1                 ! standard scheme or log transformation

  real(dp) :: &
    G = 1.0_dp,          & ! Polymer modulus
    betav = 0.59_dp,     & ! viscosity ratio
    mobility = 0.01_dp,  & ! mobility parameter in the Giesekus model (alpha)
    lambda = 1._dp         ! Polymer relaxation time for each mode

  real(dp), parameter :: &
    time0 = 0._dp,                     & ! initial time
    U_avg = 1.0_dp,       & ! average velocity at the entry
    beta = 1.0_dp,        & ! upwinding parameter in the SUPG method
    betai = 0.5_dp,       & ! upwinding parameter in the SUPG method free surface
    H = 1.0_dp,           & ! half height domain
    flowrate = H*U_avg,   & ! flowrate in half the channel
    rmin = 0.1_dp,        & ! used in refinment fields
    rs_up = 2.5_dp,       & ! real_storage gradient-velocity-pressure LU (HSL)
    is_up = 2.5_dp,       & ! integer_storage gradient-velocity-pressure LU (HSL)
    rs_c  = 4.5_dp,       & ! real_storage for the conformation LU (HSL)
    is_c  = 4.5_dp          ! integer_storage for conformation LU (HSL)

  real(dp) :: &
    dx_box = 0.4_dp,      & ! element spacing on the external boundaries
    dx_wall = 0.08_dp,    & ! element spacing on the upper boundary and die exit
    dx_inlet = 0.1_dp,    & ! element spacing on the inlet
    deltat = 1.e-2_dp       ! time step

  real(dp) :: eta_s

  ! variables

  real(dp) :: &
    lx_in = 1.0_dp,      & ! size in x-direction
    ly_in = 1.0_dp,      & ! size in y-direction
    ox_in = -4.0_dp,     & ! x coordinate lower left corner
    oy_in = 0.0_dp,      & ! y coordinate lower left corner
    rs_up_in = 2.5_dp,   & ! real_storage gradient-velocity-pressure LU (HSL)
    is_up_in = 2.0_dp      ! integer_storage gradient-velocity-pressure LU (HSL)

  integer :: icomp, step, i, m, k, obj_ob, obj_sh, obj_inlet, obj_cinlet
  integer :: nnodes_inlet, nnodes_cinlet, ipost=0, ipost_in=0
  real(dp) :: alpha_devssg

  real(dp), allocatable, dimension(:,:) :: coor
  real(dp) :: initial_h
  real(dp) :: ox=3.0_dp, L2=5.0_dp, oy=1.0_dp

  real(dp), allocatable, dimension(:,:) :: meshcoor_n, meshcoor_nm1

  character(len=300) :: filename

  logical :: devss = .true., lin_elem=.false.

  namelist /comppar/ lambda,  betav, mobility, deltat, numtimesteps, &
     dx_box, dx_wall, dx_inlet

  ! call execute_command_line ( 'rm *.vtk' )

  read ( unit=*, nml=comppar )

! set some parameters

  eta_s = betav/(1-betav)*lambda*G

  if ( devss ) then
    physqgrad = 1
    physqvel = 2
    physqpress = 3
    alpha_devssg = G*lambda  ! DEVSS parameter
  else
    physqvel = 1
    physqpress = 2
  end if

! fill coefficients

  call create_coefficients ( coefficients, ncoefi=600, ncoefr=600 )

  coefficients%i = 0

  coefficients%i = &
    [ uintpl,   pintpl,     0,     0,         gintpl, &
      physqvel, physqpress, 0,     0, gauss,  &
      gaussb,    cintpl,     0,     0,         0,      &
      0,        0,          model, nmodes,    startm, &
      logc,     0,          coorsys,  ( 0, i = 24, 600 )  &
    ]

  coefficients%i(29:31) = [ htype, 0, Uscaling ]
  coefficients%i(35) = ointpl
  coefficients%i(48) = 1 ! use mesh velocity for ALE formulation
  coefficients%i(49) = 1 ! exp(s) projection=.true. for logc=1

  coefficients%r = 0
  coefficients%r(1:10) = &
    [ eta_s, 0._dp,  0._dp, 0._dp, 0._dp, &
      0._dp, 0._dp, deltat,  beta, 0._dp  &
    ]

  coefficients%r(51:50+2*nmodes) = [ G, lambda ]
  coefficients%r(53:52+nmodes) = [mobility]

  if ( devss ) then
    coefficients%r(4) = alpha_devssg
    coefficients%i(9) = physqgrad
  else
    coefficients%i(60) = 1 ! Separate vector for velocity gradient
  end if

! fill coefficients channel

  call create_coefficients ( coefficients_inlet, ncoefi=450, ncoefr=400 )

  coefficients_inlet%i = 0

  coefficients_inlet%i = &
    (/ uintpl,   pintpl,     0,     0,         gintpl, &
       physqvel, physqpress, 0,     0, gaussb, &
       gaussb,    cintpl,     0,     0,         0,      &
       0,        0,          model, nmodes,    startm, &
       logc,     timeint1,    coorsys, ( 0, i = 24, 450 )  &
    /)


  coefficients_inlet%r = 0

  coefficients_inlet%r(1:10) = &
    [ eta_s, 0._dp,   0._dp,   0._dp, 0._dp, &
       0._dp, 0._dp,  deltat,   beta,  0._dp &
      ]

  coefficients_inlet%r(51:50+2*nmodes) = [ G, lambda ]
  coefficients_inlet%r(53:52+nmodes) = [mobility]

  coefficients_inlet%i(49) = 1 ! exp(s) projection=.true. for logc=1

  if ( devss ) then
    coefficients_inlet%r(4) = alpha_devssg
    coefficients_inlet%i(9) = physqgrad
  else
    coefficients_inlet%i(60) = 1 ! Separate vector for velocity gradient
  end if

  call generate_read_mesh

  coefficients_inlet%r(6) = flowrate

  call define_inlet_problem

  call define_problems

  call define_free_surface_problem

  call define_conformation_problem

! create the structure oldvectors_ve

  call create_oldvectors ( oldvectors_ve, nsysvec=3, nsysvec2=3, nprob=3, &
    nvec=3 )
    call create_vector ( problem, velocity, physq=physqvel )
    call create_vector ( problem, pressure, vec=4 )

! store solution vectors and problem structures

  oldvectors_ve%s(1)%p => sol
  oldvectors_ve%s2(1)%p => solcn
  oldvectors_ve%s2(2)%p => solcm1
  oldvectors_ve%s2(3)%p => solc_projcn
  oldvectors_ve%p(3)%p => problemc_projc
  oldvectors_ve%p(2)%p => problemc
  if ( .not. devss ) then
    oldvectors_ve%v(3)%p => gradients
  end if
  oldvectors_ve%p(1)%p => problem
  oldvectors_ve%v(1)%p => meshvel

  if ( .not. devss ) then

!   create gradient vector
    call create ( problem, gradients, vec=3 )

!   define gradient projection problem
    call proj_grad_definition

  end if

  coefficients%i(22) = timeint1  ! first step integration scheme
  coefficients_inlet%i(22) = timeint1  ! first step integration scheme
  coefficients_sf_adv%i(5) = 1 ! start with first-order scheme

  open(unit=11, file='snapshots_add.txt', status='replace')
  ! open(unit=14, file='pressure_add.txt', status='replace')
  open(unit=15, file='mesh_coor2_add.txt', status='replace')
  open(unit=40, file='c-trace_add.txt', status='replace')

  do step = 1, numtimesteps

    if ( step == 2 ) then
      coefficients%i(22) = timeint2
      coefficients_inlet%i(22) = timeint2
      coefficients_sf_adv%i(5) = 2
    end if

    call solve_inlet_problem

    if (step == 1) then
      write(15, *) mesh%coor(:,2)
    end if

    if ( step >= 2 ) then

      coefficients_sf_adv%i(5) = 2  ! second-order scheme

!     predict position of the surface and adapt mesh accordingly
      Hhatn = Hhat
      if ( .not. lin_elem ) then
        Hhat = 2._dp*sol_sf_adv_n%u(subsh%s) - sol_sf_adv_nm1%u(subsh%s)
      end if
      sol_sf_adv_pred%u = 2._dp*sol_sf_adv_n%u(subsh%s) - sol_sf_adv_nm1%u(subsh%s)

      if ( lin_elem ) then

        call derive_vector ( mesh_sf_adv, problem_sf_adv, height_sf_adv, &
          elemsub=height_function_deriv, &
          coefficients=coefficients_sf_adv, oldvectors=oldvectors_sf_adv_deriv )
       
        Hhat = height_sf_adv%u

      end if

      meshcoor_nm1 = meshcoor_n
      meshcoor_n = mesh%coor

      call update_mesh_nodes_2D ( mesh, problem_lapl, disp=Hhat-Hhatn )
      write(15, *) mesh%coor(:,2)

      call find_bounds_blocks ( mesh )

!   mesh velocity

      meshvel%u = reshape ( transpose ( &
          ( 1.5_dp*mesh%coor - 2*meshcoor_n + 0.5_dp*meshcoor_nm1 ) / deltat ), &
                           [2*mesh%nnodes] )

    end if

!   sample velocity and conformation tensor and use them as boundary conditions
!   at the inlet

    nnodes_inlet = mesh%curves(6)%nnodes
    mesh_inlet%objects(obj_inlet)%coor(1:nnodes_inlet,1) = &
      mesh%coor(mesh%curves(6)%nodes,1)
    mesh_inlet%objects(obj_inlet)%coor(1:nnodes_inlet,2) = &
      mesh%coor(mesh%curves(6)%nodes,2)
    call fill_sample ( mesh_inlet, problem_inlet, sample_v, &
      ndegfd=ndim, object=obj_inlet, &
      elemsub=stokes_sample_velocity, coefficients=coefficients_inlet, &
      oldvectors=oldvectors_ve_inlet )
    vel_inlet(nnodes_inlet,:) = sample_v%u(nnodes_inlet,:)
 
    mesh_inlet%objects(obj_cinlet)%coor(1:nnodes_cinlet,1) = &
      mesh%coor(cc_in(1)%nodes(:),1)
    mesh_inlet%objects(obj_cinlet)%coor(1:nnodes_cinlet,2) = &
      mesh%coor(cc_in(1)%nodes(:),2)
    call fill_sample ( mesh_inlet, problemc_inlet, sample_c, &
      ndegfd=ncompc, object=obj_cinlet, &
      elemsub=sample_conformation_tensor_std, coefficients=coefficients_inlet, &
      oldvectors=oldvectors_ve_inlet )
    
!   fill solution vector with essential boundary conditions

    do i = 1, size( cc_in(1)%s(:) )
      do m=1,nmodes
        do icomp = 1, ncompc
          solc(icomp,m)%u(cc_in(1)%s(i)) = sample_c%u(i,icomp)
        end do
      end do
    end do

!   exps projection ( for log conformation )
    if ( logc == 1 ) then
       call solve_exps_projection
    end if

!   fill solution vector with essential boundary conditions

    do i = 1, ndim
      sol%u(vel_in(i)%s) = sample_v%u(:,i)
    end do

!   build (assemble) matrix/vector for gradient/velocity/pressure problem

    call build_vpG

!   build implicit terms of CE

    call build_system ( mesh, problem, sysmatrix, rhsd, &
      elemsub=divtau_implicit_ce_elem_c, &
      oldvectors=oldvectors_ve, physqrow=[physqvel], physqcol=[physqvel], &
      addmatvec=.true., coefficients=coefficients )

    call check ( sysmatrix )

    call add_effect_of_essential_to_rhs ( problem, sysmatrix, sol, rhsd )

!      solve gradient/velocity/pressure problem

     if (step == 1 ) then
        solver_options_u%pivot_order = 1
        solver_options_u%scaling = 1
        call renumber_metis_sysmatrix(sysmatrix)
     end if

    solver_options_u%real_storage=rs_up
    solver_options_u%integer_storage=is_up

    call solve_system_ma41 ( sysmatrix, rhsd, sol, &
      solver_options=solver_options_u  )

    if ( .not. devss ) call build_and_solve_proj_grad

!   build (assemble) matrix and vector for conformation problem   

    if ( coefficients%i(22) == timeint1 ) then
      
      call build_system ( mesh, problemc, sysmatrixc, m2sysvector=rhsc, &
        elemsub=ce_supg_elem1, oldvectors=oldvectors_ve, &
        coefficients=coefficients )   
      
    else

      call build_system ( mesh, problemc, sysmatrixc, m2sysvector=rhsc, &
        elemsub=ce_supg_elem_implicit_2nd_order, oldvectors=oldvectors_ve, &
        coefficients=coefficients )
    
    end if

    call check ( sysmatrixc )

    do m=1,nmodes
      do icomp = 1, ncompc
        call add_effect_of_essential_to_rhs ( problemc, sysmatrixc, solc(icomp,m), &
          rhsc(icomp,m) )
      end do

!     solve conformation and keep LU decomposition in the loop over components

     if (step == 1) then
        solver_options_c%pivot_order = 1
        solver_options_c%scaling = 1
        call renumber_metis_sysmatrix(sysmatrixc)
     end if

      solver_options_c%real_storage=rs_c
      solver_options_c%integer_storage=is_c

      do icomp = 1, ncompc
        call solve_system_ma41 ( sysmatrixc, rhsc(icomp,m), solc(icomp,m), luc, &
          solver_options=solver_options_c  )
      end do

      call delete ( luc )  ! remove LU decomposition and rebuild next time step

    call extract_physvector ( mesh, problem, sol, velocity )
    write(11, *) velocity%u

    call derive_vector ( mesh, problem, pressure, &
      elemsub=stokes_pressure, coefficients=coefficients, &
      oldvectors=oldvectors_ve )
    ! write(14, *) pressure%u

    end do

!   solve surface advection (corrector)

    call solve_surface_height_corrector

!   create a vector for conformation tensor
!   tensor for post processing

    call create_vector ( problemc, ctensor, vec=3 )

    call derive_vector ( mesh, problemc, ctensor, &
      elemsub=deriv_conformation_tensor, &
      coefficients=coefficients, oldvectors=oldvectors_ve )

    write(40, fmt=*) ctensor%u(cxx%s) + ctensor%u(cyy%s)

    if ( step == 1 ) then
      open(unit=12, recl=300, status='replace', file='cval.out')
    else
      open(unit=12, recl=300, position='append', file='cval.out')
    end if
    write(12, fmt=*) step * deltat, maxval(ctensor%u(cxx%s)), &
                                    maxval(ctensor%u(cxy%s)), &
                                    maxval(ctensor%u(cyy%s)), &
                                    sum(ctensor%u(cxx%s))/size(cxx%s), &
                                    sum(ctensor%u(cxy%s))/size(cxy%s), &
                                    sum(ctensor%u(cyy%s))/size(cyy%s)
    close(unit=12)

    call delete ( ctensor )

    if (step ==1 ) then
      call postprocessing 
      ipost = ipost + 1
    end if

!   write VTK files
    if ( vtkevery > 0 ) then
      if ( mod(step,vtkevery) == 0 ) then
        call postprocessing 
        ipost = ipost + 1
      end if
    end if

!   copy old values

    call copy ( soln, solm1 )
    call copy ( sol, soln )

    call copy ( solcn, solcm1 )
    call copy ( solc, solcn )
    call copy ( solc_projcn, solc_projcm1 )
  
  end do

  close(11)
  close(14)
  close(15)
  close(40)

! delete all data including all allocated memory

  call delete ( mesh )
  call delete ( problem )
  call delete ( input_probdef )
  call delete ( sol, soln, solm1, rhsd )
  call delete ( sysmatrix )
  call delete ( coefficients )
  call delete ( oldvectors_ve )

  call delete ( problemc )
  call delete ( input_probdefc )
  call delete ( sysmatrixc )
  call delete ( solc, solcn, solcm1, rhsc )
  call delete ( solc_projcn, solc_projcm1 )

  call delete ( meshvel )

  deallocate ( meshcoor_initial, meshcoor_n, meshcoor_nm1 )

  call delete ( mesh_sf_adv )
  call delete ( problem_sf_adv )
  call delete ( input_probdef_sf_adv )
  call delete ( sol_sf_adv, rhsd_sf_adv )
  call delete ( sol_sf_adv_pred, sol_sf_adv_n, sol_sf_adv_nm1 )
  call delete ( sysmatrix_sf_adv )
  call delete ( oldvectors_sf_adv )
  call delete ( coefficients_sf_adv )
  call delete ( hgt, hgt_end )
  call delete ( problem_lapl)

  call delete ( mesh_inlet )
  call delete ( problem_inlet )
  call delete ( input_probdef_inlet )
  call delete ( sol_inlet, soln_inlet, solm1_inlet, rhsd_inlet )
  call delete ( sysmatrix_inlet )
  call delete ( coefficients_inlet )
  call delete ( oldvectors_ve_inlet )
  call delete ( problemc_inlet )
  call delete ( input_probdefc_inlet )
  call delete ( sysmatrixc_inlet )
  call delete ( solc_inlet, solcn_inlet, solcm1_inlet, rhsc_inlet )
  call delete ( solc_projcn_inlet, solc_projcm1_inlet )

  if ( .not. devss ) then
    call delete ( gradients )
    call delete ( input_probdef_grad )
    call delete ( problem_grad )
    call delete ( sysmatrix_grad )
    call delete ( sol_grad )
    call delete ( rhsd_grad )
    call delete ( oldvectors_grad )
    call delete ( gradients_inlet )
    call delete ( input_probdef_grad_inlet )
    call delete ( problem_grad_inlet )
    call delete ( sysmatrix_grad_inlet )
    call delete ( sol_grad_inlet )
    call delete ( rhsd_grad_inlet )
    call delete ( oldvectors_grad_inlet )
  end if

contains

! generate and read mesh
  subroutine generate_read_mesh
    logical :: mesh_exists

    inquire ( file='mesh.msh', exist=mesh_exists )

    if ( .not. mesh_exists ) then
      call write_gmsh_parameters ( ox, oy, L2, dx_box, dx_wall, dx_inlet )
      call execute_command_line ( 'gmsh -2 -order 2 -algo front2d -o mesh.msh &
                    &mesh.geo > outputmesh.out' )
    end if

!   read mesh generated by gmsh
    call read_mesh_gmsh ( mesh, filename='mesh.msh', ndim=2, &
      physgeom=.true. )

    call add_to_mesh ( mesh, object='curve', objectcurve=6, topology=.true., &
      intrule=3 )
    obj_ob = mesh%nobjects

    call add_to_mesh ( mesh, curve=[4,5] )
    
    call fill_mesh_parts ( mesh )

    call printinfo ( mesh, printlevel=2 )

!   define some arrays for the ALE mesh position at old times
    allocate ( meshcoor_n(mesh%nnodes,mesh%ndim), &
      meshcoor_nm1(mesh%nnodes,mesh%ndim) )

    allocate ( meshcoor_initial(mesh%nnodes,2) )
      meshcoor_initial = mesh%coor

    meshcoor_n = mesh%coor
    meshcoor_nm1 = mesh%coor

    call write_mesh_vtk ( mesh, filename='mesh.vtk' )

  end subroutine generate_read_mesh

! write the parameters in gmsh format

  subroutine write_gmsh_parameters ( ox, oy, L2, dx_box, dx_wall, dx_inlet )

    real(dp) :: ox, oy, L2, dx_box, dx_wall, dx_inlet

  call add_refinement_field ( refinement_fields, coor=reshape([0._dp,H],[1,2]), &
    distmin=4*rmin, distmax=8*rmin, dx_fine=dx_wall, dx_coarse=dx_box )

    open ( unit=25, file='mesh.geo' )

    write ( 25, '(1X,A,F18.14,A)' ) 'ox = ', ox, ';'
    write ( 25, '(1X,A,F18.14,A)' ) 'oy = ', oy, ';'
    write ( 25, '(1X,A,F18.14,A)' ) 'L2 = ', L2, ';'
    write ( 25, '(1X,A,F18.14,A)' ) 'dx_box = ', dx_box, ';'

    write ( 25, '(1X,A,F18.14,A)' ) 'dx_wall = ', dx_wall, ';'
    write ( 25, '(1X,A,F18.14,A)' ) 'dx_inlet = ', dx_inlet, ';'

!   write the refinement fields (the file needs to be open for writing)
    call write_refinement_fields ( refinement_fields, 'mesh.geo' )

    write ( 25, '(/1x,a)' ) 'Include "mesh_2D_bc.igo";'
    close ( 25 )

!   refinement points served their purpose
    call delete_refinement_fields ( refinement_fields )

  end subroutine write_gmsh_parameters

  subroutine build_vpG

!   build (assemble) matrix and vector for gradient/velocity/pressure problem
  
!     stokes velocity/pressure
      call build_system ( mesh, problem, sysmatrix, rhsd, &
        elemsub=stokes_elem, physqrow=[physqvel,physqpress], &
        physqcol=[physqvel,physqpress], coefficients=coefficients )

      if (devss ) then
!       DEVSS-G
        call build_system ( mesh, problem, sysmatrix, rhsd, &
          elemsub=devssg_elem, addmatvec=.true., &
          physqrow=[1,2], physqcol=[1,2], coefficients=coefficients )

!       set to zero off-diagonal blocks gradient-pressure
        call build_system ( mesh, problem, sysmatrix, rhsd, addmatvec=.true., &
          buildvector=.false., physqrow=[1], physqcol=[3], zeromatvec=.true. )
        call build_system ( mesh, problem, sysmatrix, rhsd, addmatvec=.true., &
          buildvector=.false., physqrow=[3], physqcol=[1], zeromatvec=.true. )
      end if
  
  end subroutine build_vpG

! solve convection equation for the surface height (corrector)

  subroutine solve_surface_height_corrector

    use postprocessing_m

    real(dp) :: max_height, end_height
    type(solver_options_ma41_t) :: solver_options_h

    velocity_sf_adv%u(velx_sf%s) = sol%u(velx7%s)
    velocity_sf_adv%u(vely_sf%s) = sol%u(vely7%s)

    call build_system ( mesh_sf_adv, problem_sf_adv, sysmatrix_sf_adv, &
      rhsd_sf_adv, elemsub=surface_advection_elem, &
      oldvectors=oldvectors_sf_adv, coefficients=coefficients_sf_adv )

    call check_filled_sysmatrix ( sysmatrix_sf_adv )

    call add_effect_of_essential_to_rhs ( problem_sf_adv, sysmatrix_sf_adv, &
      sol_sf_adv, rhsd_sf_adv )

!   MA41 solver storage 

   if (step == 1) then
      solver_options_h%pivot_order = 1
      solver_options_h%scaling = 1
      call renumber_metis_sysmatrix(sysmatrix_sf_adv)
   end if

    solver_options_h%integer_storage = 2.0
    solver_options_h%real_storage    = 2.0

!   solve system

    call solve_system_ma41 ( sysmatrix_sf_adv, rhsd_sf_adv, sol_sf_adv, &
                             solver_options=solver_options_h )


    call copy ( sol_sf_adv_n, sol_sf_adv_nm1 )
    call copy ( sol_sf_adv, sol_sf_adv_n )

!   compute the maximum, minimum and end radius

    max_height = maxval ( sol_sf_adv%u )
    end_height = sol_sf_adv%u(hgt_end%s(1))

  print '(i6,4es16.8)', step0+step, time0+step*deltat, &
                          max_height, end_height, max_height/initial_h

  end subroutine solve_surface_height_corrector

  subroutine postprocessing 

    type(oldvectors_t) :: oldvectors_dve
    type(vector_t) :: pressure, eff_shear

    if ( .not. mesh%meshparts) call fill_mesh_parts ( mesh )

    call write_mesh_vtk ( mesh, filename='mesh.vtk' )

    call create_oldvectors ( oldvectors_dve, nsysvec=1, nsysvec2=1 )
    oldvectors_dve%s(1)%p => sol

    call create_vector ( problem, pressure, vec=4 )

!   derive the pressure in all nodes
    call derive_vector ( mesh, problem, pressure, &
      elemsub=stokes_pressure, coefficients=coefficients, &
      oldvectors=oldvectors_dve )

    call create_vector ( problem, eff_shear, vec=4 )

    coefficients%i(13) = 8

!   derive the effective shear rate in all nodes
    call derive_vector ( mesh, problem, eff_shear, &
      elemsub=stokes_deriv, coefficients=coefficients, &
      oldvectors=oldvectors_dve )

    write(filename,'(a,i4.4,a)') 'flow', ipost, '.vtk'
    call write_scalar_vtk ( mesh, problem, vector=pressure, &
      dataname='pressure',  filename=filename )

    call write_vector_vtk ( mesh, problem, filename=filename, &
      dataname='velocity', sysvector=sol, physq=physqvel, &
      append=.true. )

    write(filename,'(a,i4.4,a)') 'shearrate', ipost, '.vtk'
    call write_scalar_vtk ( mesh, problem, vector=eff_shear, &
      dataname='effective_shearrate',  filename=filename )

    call create_vector ( problemc, ctensor, vec=3 )

    call derive_vector ( mesh, problemc, ctensor, &
      elemsub=deriv_conformation_tensor, &
      coefficients=coefficients, oldvectors=oldvectors_ve )

    write(filename,'(a,i4.4,a)') 'c_', ipost, '.vtk'
    call write_tensor_vtk ( mesh, problemc, filename=filename, &
      dataname='conformation_tensor', vector=ctensor, append=.true. )

    call printtofile ( mesh, problemc, filename='conformation_cl.out', curve=7, &
      vector=ctensor )

    call delete ( ctensor )

    call delete(pressure)
    call delete(oldvectors_dve)

    end subroutine postprocessing

    subroutine define_inlet_problem

     !  create mesh channel

      open ( unit=25, file='mesh_inlet.geo' )

      write ( 25, '(1X,A,F18.14,A)' ) 'ox_in = ', ox_in, ';'
      write ( 25, '(1X,A,F18.14,A)' ) 'oy_in = ', oy_in, ';'
      write ( 25, '(1X,A,F18.14,A)' ) 'lx_in = ', lx_in, ';'
      write ( 25, '(1X,A,F18.14,A)' ) 'ly_in = ', ly_in, ';'
      write ( 25, '(1X,A,F18.14,A)' ) 'dx_box = ', dx_box, ';'

      write ( 25, '(1X,A,F18.14,A)' ) 'dx_wall = ', dx_wall, ';'
      write ( 25, '(1X,A,F18.14,A)' ) 'dx_inlet = ', dx_inlet, ';'

      write ( 25, '(/1x,a)' ) 'Include "mesh_inlet_2D_bc.igo";'
      close ( 25 )

      call execute_command_line ( 'gmsh -2 -order 2 -algo front2d -o mesh_inlet.msh &
                  &mesh_inlet.geo > outputmesh_inlet.out' )
  
!     read mesh generated by gmsh
      call read_mesh_gmsh ( mesh_inlet, filename='mesh_inlet.msh', ndim=2, &
        physgeom=.true. )

      call add_to_mesh ( mesh_inlet, matchingcurve=[4,2], replace=4, &
         displacement=[-1._dp,0._dp] )

      call add_to_mesh ( mesh_inlet, curve=(/-4/) )     ! curve 5

      call fill_mesh_parts ( mesh_inlet )

!    problem definition of gradient/velocity/pressure for the channel

     if ( devss ) then
  
       call create_input_probdef ( mesh_inlet, input_probdef_inlet, nvec=4, nphysq=3 )

       input_probdef_inlet%vec_elementdof(1)%a =   &
           reshape ( (/ 4,0,4,0,4,0,    &  ! G
                        2,2,2,2,2,2,    &  ! velocity
                        1,0,1,0,1,0,    &  ! pressure
                        1,1,1,1,1,1 /), &  ! scalar, such as vorticity
                        (/6,4/) )

       input_probdef_inlet%physq = (/1,2,3/)
       input_probdef_inlet%probnr = 1
     else
       call create_input_probdef ( mesh_inlet, input_probdef_inlet, nvec=4, nphysq=2 )

       input_probdef_inlet%vec_elementdof(1)%a =   &
           reshape ( (/ 2,2,2,2,2,2,    &  ! velocity
                        1,0,1,0,1,0,    &  ! pressure
                        4,0,4,0,4,0,    &  ! G
                        1,1,1,1,1,1 /), &  ! scalar, such as vorticity
                        (/6,4/) )

       input_probdef_inlet%physq = (/physqvel, physqpress/)
       input_probdef_inlet%probnr = 1
     end if

!   Dirichlet boundary conditions

!   velocity on lower boundary
    call define_essential ( mesh_inlet, input_probdef_inlet, curve1=1, physq=physqvel, &
      degfd=[0,1], excludecurves=(/4/) )
!   velocity on upper boundary
    call define_essential ( mesh_inlet, input_probdef_inlet, curve1=3, physq=physqvel, &
        excludecurves=(/4/) )
!   pressure in point 1 
    call define_essential ( mesh_inlet, input_probdef_inlet, point=1, physq=physqpress )

!   constraints for periodical boundary conditions 

!   velocities
    call define_constraint ( mesh_inlet, input_probdef_inlet, &
      physq=physqvel, curve1=2, curve2=4, discretization='collocation')

!   constraint for the flow rate
    call define_constraint ( mesh_inlet, input_probdef_inlet, &
      physq=physqvel, curve1=4, nglobalc=1 )

    call problem_definition ( input_probdef_inlet, mesh_inlet, problem_inlet )

!   problem definition conformation tensor

    call create_input_probdef ( mesh_inlet, input_probdefc_inlet, nvec=2, nphysq=1 )

    input_probdefc_inlet%vec_elementdof(1)%a =   &
        reshape ( (/ 1,0,1,0,1,0,    &  ! c 
                     1,1,1,1,1,1 /), &  ! scalar for plotting
                     (/6,2/) )
  
    input_probdefc_inlet%physq = (/1/)
    input_probdefc_inlet%probnr = 2

!   constraint for periodical boundary conditions of the conformation

    call define_constraint ( mesh_inlet, input_probdefc_inlet, curve1=2, curve2=4, &
      discretization='collocation' )

    call problem_definition ( input_probdefc_inlet, mesh_inlet,  problemc_inlet )

!   create system vectors for gradient/velocity/pressure 
!   (solution and right-hand side) for the channel problem

    call create_sysvector ( problem_inlet, sol_inlet, soln_inlet, solm1_inlet )
    call create_sysvector ( problem_inlet, rhsd_inlet )

    sol_inlet%u = 0._dp
    solm1_inlet%u = 0._dp

!   fill solution vector with essential boundary conditions

!   set y velocity = 0 on axis
    call fill_sysvector ( mesh_inlet, problem_inlet, sol_inlet, &
      curve1=1, physq=physqvel, degfd=2, value=0._dp )
!   set velocity = 0 on upper boundary
    call fill_sysvector ( mesh_inlet, problem_inlet, sol_inlet, &
      curve1=3, physq=physqvel, degfd=1, value=0._dp )
    call fill_sysvector ( mesh_inlet, problem_inlet, sol_inlet, &
      curve1=3, physq=physqvel, degfd=2, value=0._dp )
!   set pressure level = 0 in lower left corner
    call fill_sysvector ( mesh_inlet, problem_inlet, sol_inlet, &
      point=1, physq=physqpress, value=0._dp )

!   create system matrix of gradient/velocity/pressure problem

    call create_sysmatrix_structure_base ( sysmatrix_inlet, mesh_inlet, problem_inlet )
    call create_sysmatrix_structure_constraint ( sysmatrix_inlet, mesh_inlet, problem_inlet )
    call finalize_sysmatrix_structure ( sysmatrix_inlet )

    call create_sysmatrix_data ( sysmatrix_inlet )

!   create system vectors (solution and right-hand side) for conformation and
!   initialize vectors with zero stress for the channel

    call create ( problemc_inlet, solc_inlet, solcn_inlet, solcm1_inlet, &
                rhsc_inlet )

    if ( logc == 0 ) then ! standard
      solc_inlet(1,1)%u = 1   ! initial cxx
      solc_inlet(2,1)%u = 0   ! initial cxy
      solc_inlet(3,1)%u = 1   ! initial cyy
    else if ( logc == 1 ) then ! log scheme
      solc_inlet(1,1)%u = 0   ! initial sxx
      solc_inlet(2,1)%u = 0   ! initial sxy
      solc_inlet(3,1)%u = 0   ! initial syy
    end if

    call copy ( solc_inlet, solcm1_inlet )
    call copy ( solc_inlet, solcn_inlet )

!   create system matrix for conformation problem

    call create_sysmatrix_structure_base ( sysmatrixc_inlet, mesh_inlet, &
      problemc_inlet )
    call create_sysmatrix_structure_constraint ( sysmatrixc_inlet, mesh_inlet, &
      problemc_inlet )
    call finalize_sysmatrix_structure ( sysmatrixc_inlet )

    call create_sysmatrix_data ( sysmatrixc_inlet )

!   problem definition for projected "c=exp(s)" of the log conformation s
    call create_input_probdef ( mesh_inlet, input_probdefc_projc_inlet, nvec=1, &
      nphysq=1 )

      input_probdefc_projc_inlet%vec_elementdof(1)%a = &
          reshape ( [ 1,0,1,0,1,0 ], &
                         [6,1] )

     input_probdefc_projc_inlet%physq = [1]
     input_probdefc_projc_inlet%probnr = 3

    call problem_definition ( input_probdefc_projc_inlet, mesh_inlet, problemc_projc_inlet )

    call create ( problemc_projc_inlet, solc_projcn_inlet, &
        solc_projcm1_inlet, rhsc_projc_inlet )

    solc_projcn_inlet(1,1)%u = 1   ! initial cxx
    solc_projcn_inlet(2,1)%u = 0   ! initial cxy
    solc_projcn_inlet(3,1)%u = 1   ! initial cyy

!   create and build system matrix for projection problem
!   NOTE matrix remains constant and needs to be build once.

    call create_sysmatrix_structure ( sysmatrixc_projc_inlet, mesh_inlet, &
                                   problemc_projc_inlet)!, symmetric=.true. )
    call create_sysmatrix_data ( sysmatrixc_projc_inlet )
!
    call build_system ( mesh_inlet, problemc_projc_inlet, sysmatrixc_projc_inlet, &
      m2sysvector=rhsc_projc_inlet, elemsub=exps_projection_elem, &
      oldvectors=oldvectors_ve_inlet, coefficients=coefficients_inlet, &
      buildvector=.false. )

    call check ( sysmatrixc_projc_inlet )

!   create the structure oldvectors_ve for the channel problem

    call create_oldvectors ( oldvectors_ve_inlet, nsysvec=5, nsysvec2=3, nprob=3, &
         nvec=3 )

!   store solution vectors and problem structures

    oldvectors_ve_inlet%s(1)%p => sol_inlet
    oldvectors_ve_inlet%p(1)%p => problem_inlet
    oldvectors_ve_inlet%p(2)%p => problemc_inlet
    oldvectors_ve_inlet%s2(1)%p => solcn_inlet
    oldvectors_ve_inlet%s2(2)%p => solcm1_inlet
    oldvectors_ve_inlet%s2(3)%p => solc_projcn_inlet
    oldvectors_ve_inlet%p(3)%p => problemc_projc_inlet

    if ( .not. devss ) then
      oldvectors_ve_inlet%v(3)%p => gradients_inlet

!     create gradient vector
      call create ( problem_inlet, gradients_inlet, vec=3)

!     define gradient projection problem
      call proj_grad_definition_inlet

    end if

  end subroutine define_inlet_problem 

  subroutine solve_inlet_problem

!   exps projection ( for log conformation )
    if ( logc == 1 ) then
       call solve_exps_projection_inlet
    end if

!   build (assemble) matrix/vector for gradient/velocity/pressure problem 
!   for channel problem

    call build_vpG_inlet

!   build implicit terms of CE with rhs in momentum balance

    call build_system ( mesh_inlet, problem_inlet, sysmatrix_inlet, rhsd_inlet, &
      elemsub=divtau_implicit_ce_elem_c, &
      oldvectors=oldvectors_ve_inlet, physqrow=(/physqvel/), physqcol=(/physqvel/), &
      addmatvec=.true., coefficients=coefficients_inlet )

    call add_effect_of_essential_to_rhs ( problem_inlet, sysmatrix_inlet, &
      sol_inlet, rhsd_inlet )

!   solve gradient/velocity/pressure problem for channel problem

    solver_options_u_in%real_storage=rs_up_in
    solver_options_u_in%integer_storage=is_up_in

    call solve_system_ma41 ( sysmatrix_inlet, rhsd_inlet, sol_inlet, &
      solver_options=solver_options_u_in  )

    if (.not. devss ) call build_and_solve_proj_grad_inlet

!   build (assemble) matrix and vector for conformation problem   
!   for channel problem 

    if ( coefficients_inlet%i(22) == timeint1 ) then

      call build_system ( mesh_inlet, problemc_inlet, sysmatrixc_inlet, &
        m2sysvector=rhsc_inlet, &
        elemsub=ce_supg_elem1, oldvectors=oldvectors_ve_inlet, &
        coefficients=coefficients_inlet )

    else

      call build_system ( mesh_inlet, problemc_inlet, sysmatrixc_inlet, &
        m2sysvector=rhsc_inlet, &
        elemsub=ce_supg_elem_implicit_2nd_order, oldvectors=oldvectors_ve_inlet, &
        coefficients=coefficients_inlet )

    end if

!   periodical condition on conformation tensor
    call build_system_constraint ( mesh_inlet, problemc_inlet, sysmatrixc_inlet, &
      m2sysvector=rhsc_inlet, elemsub=stokes_constr_node_conn, &
      addmatvec=.true. )

    call check ( sysmatrixc_inlet )

!   solve conformation and keep LU decomposition in the loop over components

    solver_options_c_in%real_storage=rs_c
    solver_options_c_in%integer_storage=is_c

    do m= 1, nmodes
      do icomp = 1, ncompc
        call solve_system_ma41 ( sysmatrixc_inlet, rhsc_inlet(icomp,m), &
        solc_inlet(icomp,m), luc_inlet, &
        solver_options=solver_options_c_in  )
      end do

      call delete ( luc_inlet )  ! remove LU decomposition and rebuild next time step

    end do

    if (step ==1 ) then
      call postprocessing_inlet 
      ipost_in = ipost_in + 1
    end if

!   write VTK files
    if ( vtkevery > 0 ) then
      if ( mod(step,vtkevery) == 0 ) then
         call postprocessing_inlet
         ipost_in = ipost_in + 1
      end if
    end if

!   copy old values

    call copy ( soln_inlet, solm1_inlet )
    call copy ( sol_inlet, soln_inlet )
    call copy ( solcn_inlet, solcm1_inlet )
    call copy ( solc_inlet, solcn_inlet )
    call copy ( solc_projcn_inlet, solc_projcm1_inlet )

  end subroutine solve_inlet_problem

  subroutine build_vpG_inlet

!   build (assemble) matrix/vector for velocity/pressure problem

    call build_system ( mesh_inlet, problem_inlet, sysmatrix_inlet, rhsd_inlet, &
      elemsub=stokes_elem, physqrow=(/physqvel,physqpress/), physqcol=(/physqvel,physqpress/),&
      coefficients=coefficients_inlet )
 
    if ( devss ) then

!     DEVSS-G
      call build_system ( mesh_inlet, problem_inlet, sysmatrix_inlet, rhsd_inlet, &
        elemsub=devssg_elem, addmatvec=.true., &
        physqrow=(/1,2/), physqcol=(/1,2/), coefficients=coefficients_inlet )

!     set to zero off-diagonal blocks gradient-pressure
      call build_system ( mesh_inlet, problem_inlet, sysmatrix_inlet, rhsd_inlet, &
        addmatvec=.true., &
        buildvector=.false., physqrow=(/1/), physqcol=(/3/), zeromatvec=.true. )
      call build_system ( mesh_inlet, problem_inlet, sysmatrix_inlet, &
        rhsd_inlet, addmatvec=.true., &
        buildvector=.false., physqrow=(/3/), physqcol=(/1/), zeromatvec=.true. )

    end if

!   periodical condition on velocities 

    call build_system_constraint ( mesh_inlet, problem_inlet, & 
      sysmatrix_inlet, rhsd_inlet, &
      constraint1=1, elemsub=stokes_constr_node_conn, addmatvec=.true., &
      coefficients=coefficients_inlet )

!   imposed flow rate

    call build_system_constraint ( mesh_inlet, problem_inlet, &
      sysmatrix_inlet, rhsd_inlet, &
      constraint1=2, elemsub=stokes_constr_flowr, addmatvec=.true., &
      coefficients=coefficients_inlet )

  end subroutine build_vpG_inlet

  subroutine postprocessing_inlet 

    type(oldvectors_t) :: oldvectors_dve
    type(vector_t) :: pressure, eff_shear_in

    if ( .not. mesh_inlet%meshparts) call fill_mesh_parts ( mesh_inlet )

    call write_mesh_vtk ( mesh_inlet, filename='mesh_inlet.vtk' )

    call create_oldvectors ( oldvectors_dve, nsysvec=1, nsysvec2=1 )
    oldvectors_dve%s(1)%p => sol_inlet

    call create_vector ( problem_inlet, pressure, vec=4 )

!   derive the pressure in all nodes
    call derive_vector ( mesh_inlet, problem_inlet, pressure, &
      elemsub=stokes_pressure, coefficients=coefficients_inlet, &
      oldvectors=oldvectors_dve )

    call create_vector ( problem_inlet, eff_shear_in, vec=4 )

    coefficients_inlet%i(13) = 8

!   derive the effective shear rate in all nodes
    call derive_vector ( mesh_inlet, problem_inlet, eff_shear_in, &
      elemsub=stokes_deriv, coefficients=coefficients_inlet, &
      oldvectors=oldvectors_dve )

    write(filename,'(a,i4.4,a)') 'shearrate_in', ipost, '.vtk'
    call write_scalar_vtk ( mesh_inlet, problem_inlet, vector=eff_shear_in, &
      dataname='effective_shearrate',  filename=filename )

!    coefficients_inlet%i(13) = 8

    write(filename,'(a,i4.4,a)') 'flow_in', ipost_in, '.vtk'
    call write_scalar_vtk ( mesh_inlet, problem_inlet, vector=pressure, &
      dataname='pressure',  filename=filename )

    call write_vector_vtk ( mesh_inlet, problem_inlet, filename=filename, &
      dataname='velocity', sysvector=sol_inlet, physq=physqvel, &
      append=.true. )

    call delete(pressure)

    call delete(oldvectors_dve)

  end subroutine postprocessing_inlet

  subroutine solve_exps_projection

    type(solver_options_ma41_t) :: solver_options_projc
    type(oldvectors_t) :: oldvectors_proj
    type(lu_ma41_t) :: lu_projc

    integer :: i, m

    call create_oldvectors ( oldvectors_proj, nprob=2, nsysvec2=1 )
    oldvectors_proj%s2(1)%p => solcn
    oldvectors_proj%p(2)%p => problemc

!   build system matrix and vector for projection problem

    call build_system ( mesh, problemc_projc, sysmatrixc_projc, &
      m2sysvector=rhsc_projc, elemsub=exps_projection_elem, &
      oldvectors=oldvectors_proj, coefficients=coefficients )

    call check ( sysmatrixc_projc )

!   MA41 solver storage

    solver_options_projc%integer_storage = 2.3
    solver_options_projc%real_storage    = 2.3

!   LU decomposition is done in the first loop traversing

    do m = 1, nmodes
      do i = 1, ncompc

        call add_effect_of_essential_to_rhs ( problemc_projc, &
          sysmatrixc_projc, solc_projcn(i,m), rhsc_projc(i,m) )

        call solve_system_ma41 ( sysmatrixc_projc, rhsc_projc(i,m), &
           solc_projcn(i,m), lu_projc, solver_options=solver_options_projc )

      end do
    end do

    call delete ( lu_projc )

  end subroutine solve_exps_projection

! project c=exp(s) on discrete fem space

  subroutine solve_exps_projection_inlet

    type(solver_options_ma41_t) :: solver_options_ma41
    type(oldvectors_t) :: oldvectors_proj_in
    type(lu_ma41_t) :: lu_exps_projc_inlet

    integer :: i, m

    call create_oldvectors ( oldvectors_proj_in, nprob=2, nsysvec2=1 )
    oldvectors_proj_in%s2(1)%p => solcn_inlet
    oldvectors_proj_in%p(2)%p => problemc_inlet

!   build vector only (matrix is constant)

    call build_system ( mesh_inlet, problemc_projc_inlet, sysmatrixc_projc_inlet, &
      m2sysvector=rhsc_projc_inlet, elemsub=exps_projection_elem, &
      oldvectors=oldvectors_proj_in, coefficients=coefficients_inlet, &
      buildmatrix=.false. )

    ! MA57 solver storage 
    solver_options_ma41%integer_storage = 2.3
    solver_options_ma41%real_storage    = 2.3

!   LU decomposition is done in the first call only

    do m = 1, nmodes
      do i = 1, ncompc

        call add_effect_of_essential_to_rhs ( problemc_projc_inlet, sysmatrixc_projc_inlet, &
           solc_projcn_inlet(i,m), rhsc_projc_inlet(i,m) )

        call solve_system_ma41 ( sysmatrixc_projc_inlet, rhsc_projc_inlet(i,m), &
           solc_projcn_inlet(i,m), lu_exps_projc_inlet, solver_options=solver_options_ma41 )

      end do
    end do

    call delete ( lu_exps_projc_inlet )

  end subroutine solve_exps_projection_inlet

! project the velocity gradients

  subroutine build_and_solve_proj_grad

    type(lu_ma41_t) :: lu_g_ma41
    type(solver_options_ma41_t) :: solver_options_ma41

!   build (assemble) matrix and vector from elements

    call build_system ( mesh, problem_grad, sysmatrix_grad, &
      msysvector=rhsd_grad, elemsub=gradient_from_velocity_elem, &
      coefficients=coefficients, oldvectors=oldvectors_grad )

   if (step == 1) then
     solver_options_ma41%pivot_order = 1
     solver_options_ma41%scaling = 1
     call renumber_metis_sysmatrix(sysmatrix_grad)
   end if

!   MA57 solver storage
    solver_options_ma41%integer_storage = 2.3
    solver_options_ma41%real_storage    = 2.3

!   solve

    do i = 1, ncompg

      call add_effect_of_essential_to_rhs ( problem_grad, sysmatrix_grad, &
        sol_grad, rhsd_grad(i) )

        call solve_system_ma41 ( sysmatrix_grad, rhsd_grad(i), sol_grad, &
          lu=lu_g_ma41, solver_options=solver_options_ma41 )

      call transfer_data ( mesh, problem_grad, problem, &
        sysvector1=sol_grad, vector2=gradients, degfd2=[i] )

    end do

!    solver_options_ma41%pivot_order = 0

    call delete ( lu_g_ma41 )

  end subroutine build_and_solve_proj_grad

! define problem for the projection of the velocity gradients

  subroutine proj_grad_definition_inlet

    integer, dimension(3) :: vertices=[1,3,5]

!   problem definition for gradients

    call create_input_probdef ( mesh_inlet, input_probdef_grad_inlet, nvec=2, nphysq=1 )

    input_probdef_grad_inlet%vec_elementdof(1)%a(:,1) = 0
    input_probdef_grad_inlet%vec_elementdof(1)%a(vertices,1) = 1  ! gradient component

    input_probdef_grad_inlet%physq = (/1/)
    input_probdef_grad_inlet%probnr = 4

    call problem_definition ( input_probdef_grad_inlet, mesh_inlet, problem_grad_inlet )

!   create system vectors (solution and right-hand side)

    call create ( problem_grad_inlet, sol_grad_inlet )
    call create ( problem_grad_inlet, rhsd_grad_inlet )

!   create system matrix

    call create_sysmatrix_structure ( sysmatrix_grad_inlet, mesh_inlet, &
            problem_grad_inlet )

    call create_sysmatrix_data ( sysmatrix_grad_inlet )

!   oldvectors

    call create_oldvectors ( oldvectors_grad_inlet, nsysvec=1, nprob=1 )

    oldvectors_grad_inlet%s(1)%p => sol_inlet
    oldvectors_grad_inlet%p(1)%p => problem_inlet

  end subroutine proj_grad_definition_inlet

  subroutine proj_grad_definition

    integer, dimension(3) :: vertices=[1,3,5]

!   problem definition for gradients

    call create_input_probdef ( mesh, input_probdef_grad, nvec=2, nphysq=1 )

    input_probdef_grad%vec_elementdof(1)%a(:,1) = 0
    input_probdef_grad%vec_elementdof(1)%a(vertices,1) = 1  ! gradient component

    input_probdef_grad%physq = (/1/)
    input_probdef_grad%probnr = 5

    call problem_definition ( input_probdef_grad, mesh, problem_grad )

!   create system vectors (solution and right-hand side)

    call create ( problem_grad, sol_grad )
    call create ( problem_grad, rhsd_grad )

!   create system matrix

    call create_sysmatrix_structure ( sysmatrix_grad, mesh, &
            problem_grad )

    call create_sysmatrix_data ( sysmatrix_grad )

!   oldvectors

    call create_oldvectors ( oldvectors_grad, nsysvec=1, nprob=1 )

    oldvectors_grad%s(1)%p => sol
    oldvectors_grad%p(1)%p => problem

  end subroutine proj_grad_definition

  subroutine build_and_solve_proj_grad_inlet

    type(lu_ma41_t) :: lu_g_ma41
    type(solver_options_ma41_t) :: solver_options_ma41

!   build (assemble) matrix and vector from elements

    call build_system ( mesh_inlet, problem_grad_inlet, sysmatrix_grad_inlet, &
      msysvector=rhsd_grad_inlet, elemsub=gradient_from_velocity_elem, &
      coefficients=coefficients_inlet, oldvectors=oldvectors_grad_inlet )

   if (step == 1) then
     solver_options_ma41%pivot_order = 1
     solver_options_ma41%scaling = 1
     call renumber_metis_sysmatrix(sysmatrix_grad)
   end if

!   MA41 solver storage
    solver_options_ma41%integer_storage = 2.3
    solver_options_ma41%real_storage    = 2.3

!   solve

    do i = 1, ncompg

      call add_effect_of_essential_to_rhs ( problem_grad_inlet, sysmatrix_grad_inlet, &
        sol_grad_inlet, rhsd_grad_inlet(i) )

        call solve_system_ma41 ( sysmatrix_grad_inlet, rhsd_grad_inlet(i), sol_grad_inlet, &
          lu=lu_g_ma41, solver_options=solver_options_ma41 )

      call transfer_data ( mesh_inlet, problem_grad_inlet, problem_inlet, &
        sysvector1=sol_grad_inlet, vector2=gradients_inlet, degfd2=[i] )

    end do

!    solver_options_ma41%pivot_order = 0

    call delete ( lu_g_ma41 )

  end subroutine build_and_solve_proj_grad_inlet

  subroutine define_problems

!   add an obejct in the channel case, to sample the velocity and the conformation
!   tensor at the inlet of the extrudate swell problem

    allocate ( xc(mesh%curves(6)%nnodes,2) )

    xc(:,1) = mesh%coor(mesh%curves(6)%nodes,1)
    xc(:,2) = mesh%coor(mesh%curves(6)%nodes,2)

    warn_add_to_mesh_after_meshgen_parts = .false.
    call add_to_mesh ( mesh_inlet, object='coordinates', coor=xc )
    obj_inlet = mesh_inlet%nobjects
    call fill_mesh_parts_objects ( mesh_inlet, object1=obj_inlet )
    warn_add_to_mesh_after_meshgen_parts = .true.

    deallocate ( xc )

!   problem definition of gradient/velocity/pressure

    if ( devss ) then

      call create_input_probdef ( mesh, input_probdef, nvec=5, nphysq=3 )

      input_probdef%vec_elementdof(1)%a =  &
          reshape ( [ 4,0,4,0,4,0,   &  ! G
                      2,2,2,2,2,2,   &  ! velocity
                      1,0,1,0,1,0,   &  ! pressure
                      1,1,1,1,1,1,   &  ! scalar, such as vorticity
                      3,3,3,3,3,3 ], &  ! tensor
                    [6,5] )

      input_probdef%physq = [1,2,3]
      input_probdef%probnr = 7

    else
      call create_input_probdef ( mesh, input_probdef, nvec=5, nphysq=2)

      input_probdef%vec_elementdof(1)%a =  &
          reshape ( [ 2,2,2,2,2,2,   &  ! velocity
                      1,0,1,0,1,0,   &  ! pressure
                      4,0,4,0,4,0,   &  ! G
                      1,1,1,1,1,1,   &  ! scalar, such as vorticity
                      3,3,3,3,3,3 ], &  ! tensor
                    [6,5] )

      input_probdef%physq = [physqvel, physqpress]
      input_probdef%probnr = 7
      
     end if

!   Dirichlet boundary conditions

!   inlet
    call define_essential ( mesh, input_probdef, &
      curve1=6, physq=physqvel )
!   center line
    call define_essential ( mesh, input_probdef, &
      curve1=1, physq=physqvel, degfd=[0,1] )
    call define_essential ( mesh, input_probdef, &
      curve1=2, physq=physqvel, degfd=[0,1] )
!   outflow: ur=0 
    call define_essential ( mesh, input_probdef, &
      curve1=3, physq=physqvel, degfd=[0,1] )
!   wall
    call define_essential ( mesh, input_probdef, &
      curve1=5, physq=physqvel )

    call problem_definition ( input_probdef, mesh, problem )

!   subscript for the inlet BC

    do i = 1, ndim
      call create_subscript ( mesh, problem, vel_in(i), curves=[6], &
        physqarr=[physqvel], degfd=i )
    end do

    call create_subscript ( mesh, problem, velx7, physqarr=[physqvel], &
       degfd=1, curves=(/4/), fillnodes=.true. )
    call create_subscript ( mesh, problem, vely7, physqarr=[physqvel], &
       degfd=2, curves=(/4/), fillnodes=.true. )

    allocate ( vel_inlet(mesh%curves(6)%nnodes,2) )
    allocate ( c_inlet(mesh%curves(6)%nnodes,ncompc) )

!   create system vectors for gradient/velocity/pressure 
!   (solution and right-hand side)

    call create_sysvector ( problem, sol, soln, solm1, rhsd )

!   fill solution vector with essential boundary conditions

    sol%u = 0._dp
    solm1%u = 0._dp

!   create system matrix of gradient/velocity/pressure problem

    call create_sysmatrix_structure_base ( sysmatrix, mesh, problem )
    call finalize_sysmatrix_structure ( sysmatrix )

    call create_sysmatrix_data ( sysmatrix )

!   define some arrays and vector for the ALE mesh movement

    call create_vector ( problem, meshvel, physq=physqvel )
    meshvel%u = 0._dp  ! initialize

  end subroutine define_problems

  subroutine define_free_surface_problem

    if ( lin_elem ) then
      hintpl = 2
    else
      hintpl = 6
    end if

!   fill coefficients for surface advection problem

    call create_coefficients ( coefficients_sf_adv, ncoefi=100, ncoefr=50 )

    coefficients_sf_adv%i(1:4) = [ ninti_sf_adv, 2, 0, method ]
    coefficients_sf_adv%i(5:) = 0
    coefficients_sf_adv%i(5) = 1 ! time-integration
    coefficients_sf_adv%i(6) = hintpl ! height interpolation
    coefficients_sf_adv%i(9) = 3 ! numerical table for Gauss

    coefficients_sf_adv%r = 0

    coefficients_sf_adv%r(4) = deltat
    coefficients_sf_adv%r(5) = betai

    if ( lin_elem ) then
     coefficients_sf_adv%i(10) = vintpl
     coefficients_sf_adv%i(12) = 1 ! use shapefunc for velocity for element shape
    end if

!   create mesh for surface advection

    mesh_options%elshape = 2   ! three-node line elements

    mesh_options%nx = mesh%curves(4)%nelem 

    call line1d ( mesh_sf_adv, mesh_options )

!   set coordinates

    do i = 1, mesh%curves(4)%nnodes
      mesh_sf_adv%coor(i,1) = mesh%coor(mesh%curves(4)%nodes(i),1)
    end do

    call fill_mesh_parts ( mesh_sf_adv )

!   add object for sampling velocity in height function advection

    initial_h = mesh%coor(mesh%curves(4)%nodes(1),2)

    allocate ( coor(mesh_sf_adv%nnodes,2) )

    coor(:,1) = mesh_sf_adv%coor(1:mesh_sf_adv%nnodes,1)
    coor(:,2) = initial_h

    warn_add_to_mesh_after_meshgen_parts = .false.
    call add_to_mesh ( mesh, object='coordinates', coor=coor )
    obj_sh = mesh%nobjects
    call fill_mesh_parts_objects ( mesh, object1=obj_sh )
    warn_add_to_mesh_after_meshgen_parts = .true.

    deallocate ( coor )

!   allocate arrays for the ALE displacement problem

    allocate ( Hhat(mesh_sf_adv%nnodes), Hhatn(mesh_sf_adv%nnodes) )
    Hhat = initial_h

!   problem definition for surface advection 

    if ( lin_elem ) then
      call create_input_probdef ( mesh_sf_adv, input_probdef_sf_adv, nvec=3, &
        nphysq=1 )
 
      input_probdef_sf_adv%vec_elementdof(1)%a = &
         reshape ( [ 1,0,1,    &  ! height function
                  2,2,2,    &  ! velocity
                  1,1,1 ],  &  ! height in all nodes
                 [3,3] )
  
      input_probdef_sf_adv%physq = [1]
      input_probdef_sf_adv%probnr = 10

    else
      call create_input_probdef ( mesh_sf_adv, input_probdef_sf_adv, nvec=2, &
        nphysq=1 )

      input_probdef_sf_adv%vec_elementdof(1)%a = &
          reshape ( [ 1,1,1,    &  ! height function
                      2,2,2 ],  &  ! velocity
                     [3,2] )

      input_probdef_sf_adv%physq = [1]
      input_probdef_sf_adv%probnr = 9
    end if

    call define_essential ( mesh_sf_adv, input_probdef_sf_adv, point=2, physq=1 )

    print *, 'P1 BC',  mesh_sf_adv%coor(mesh_sf_adv%points(1),:)
    print *, 'P2 BC',  mesh_sf_adv%coor(mesh_sf_adv%points(2),:)

!   define problem

    call problem_definition ( input_probdef_sf_adv, mesh_sf_adv, problem_sf_adv )

    call create_sysvector ( problem_sf_adv, sol_sf_adv, rhsd_sf_adv )
    call create_sysvector ( problem_sf_adv, sol_sf_adv_n, sol_sf_adv_nm1 )
    call create_sysvector ( problem_sf_adv, sol_sf_adv_pred, sol_sf_adv_pred_n )

!   fill solution vector with essential boundary conditions

    sol_sf_adv%u = initial_h
    sol_sf_adv_n%u = sol_sf_adv%u
    sol_sf_adv_nm1%u =  sol_sf_adv_n%u

    call fill_sysvector ( mesh_sf_adv, problem_sf_adv, sol_sf_adv, point=2, &
      physq=1, value=initial_h )

!   create system matrix

    call create_sysmatrix_structure ( sysmatrix_sf_adv, mesh_sf_adv, &
      problem_sf_adv )

    call create_sysmatrix_data ( sysmatrix_sf_adv )

!   create vectors
    call create_vector ( problem_sf_adv, velocity_sf_adv, vec=2 )
    if ( lin_elem ) then
      call create_vector ( problem_sf_adv, height_sf_adv, vec=3 )
    end if

!   subscripts for velocity

    call create_subscript ( mesh_sf_adv, problem_sf_adv, velx_sf, vec=2, degfd=1 )
    call create_subscript ( mesh_sf_adv, problem_sf_adv, vely_sf, vec=2, degfd=2 )

    if ( lin_elem ) then
      call create_oldvectors ( oldvectors_sf_adv_deriv, nsysvec=1 )
      oldvectors_sf_adv_deriv%s(1)%p => sol_sf_adv_pred ! prediction
    end if

    call create_oldvectors ( oldvectors_sf_adv, nsysvec=2, nvec=1 )

    oldvectors_sf_adv%s(1)%p => sol_sf_adv_n    ! corrector at n
    oldvectors_sf_adv%s(2)%p => sol_sf_adv_nm1  ! corrector at nm1

    oldvectors_sf_adv%v(1)%p => velocity_sf_adv ! advection velocity at np1

    call create_oldvectors ( oldvectors_sample_H, nsysvec=1 )
    oldvectors_sample_H%s(1)%p => sol_sf_adv_pred

!   create subscript for the height values

    call create ( mesh_sf_adv, problem_sf_adv, subsh )
    call create ( mesh_sf_adv, problem_sf_adv, hgt )
    call create ( mesh_sf_adv, problem_sf_adv, hgt_end, points=[1] )

  end subroutine define_free_surface_problem

  subroutine define_conformation_problem

!   problem definition conformation tensor

    call create_input_probdef ( mesh, input_probdefc, nvec=3, nphysq=1 )

    input_probdefc%vec_elementdof(1)%a = &
        reshape ( [ 1,0,1,0,1,0,   &  ! c 
                    1,1,1,1,1,1,   &  ! scalar for plotting
                    ncompc,ncompc,ncompc,ncompc,ncompc,ncompc ], &  ! tensor
                  [6,3] )

    input_probdefc%physq = [1]
    input_probdefc%probnr = 10

!   inlet B.C.
    call define_essential ( mesh, input_probdefc, &
      curve1=6, physq=1 )

    call problem_definition ( input_probdefc, mesh, problemc )

!   create system vectors (solution and right-hand side) for conformation 

    call create ( problemc, solc, solcn, solcm1, rhsc )

!   initialize vectors with zero stress

    if ( logc == 0 ) then ! standard
      solc(1,1)%u = 1   ! initial cxx
      solc(2,1)%u = 0   ! initial cxy
      solc(3,1)%u = 1   ! initial cyy
    else if ( logc == 1 ) then ! log scheme
      solc(1,1)%u = 0   ! initial sxx
      solc(2,1)%u = 0   ! initial sxy
      solc(3,1)%u = 0   ! initial syy
    end if

    call copy ( solc, solcn )
    call copy ( solc, solcm1 )

!   subscript for the inlet BC

    do i = 1, ncompc
      call create_subscript ( mesh, problemc, cc_in(i), curves=[6], &
        physqarr=[1], fillnodes=.true. )
    end do
    call create_subscript_vector ( mesh, problemc, subsc_in, curves=[6], &
      vec=2 )

    call create_subscript ( mesh, problemc, cxx, degfd=1, vec=3 )
    call create_subscript ( mesh, problemc, cxy, degfd=2, vec=3 )
    call create_subscript ( mesh, problemc, cyy, degfd=3, vec=3 )

    nnodes_cinlet = size ( cc_in(1)%nodes(:) )

    allocate ( xc(nnodes_cinlet,2) )

    xc(:,1) = mesh%coor(cc_in(1)%nodes(:),1)
    xc(:,2) = mesh%coor(cc_in(1)%nodes(:),2)

    warn_add_to_mesh_after_meshgen_parts = .false.
    call add_to_mesh ( mesh_inlet, object='coordinates', coor=xc )
    obj_cinlet = mesh_inlet%nobjects
    call fill_mesh_parts_objects ( mesh_inlet, object1=obj_cinlet )
    warn_add_to_mesh_after_meshgen_parts = .true.

    deallocate ( xc )

!   create system matrix for conformation problem

    call create_sysmatrix_structure ( sysmatrixc, mesh, problemc )
    call create_sysmatrix_data ( sysmatrixc )

!   problem definition for projected "c=exp(s)" of the log conformation s
    call create_input_probdef ( mesh, input_probdefc_projc, nvec=1, &
      nphysq=1 )

      input_probdefc_projc%vec_elementdof(1)%a = &
          reshape ( [ 1,0,1,0,1,0 ], &
                       [6,1] )

    input_probdefc_projc%physq = [1]
    input_probdefc_projc%probnr = 11

    call problem_definition ( input_probdefc_projc, mesh, problemc_projc )

    call create ( problemc_projc, solc_projcn, solc_projcm1, rhsc_projc )

    solc_projcn(1,1)%u = 1   ! initial cxx
    solc_projcn(2,1)%u = 0   ! initial cxy
    solc_projcn(3,1)%u = 1   ! initial cyy

!   create system matrix for projection problem
    call create_sysmatrix_structure ( sysmatrixc_projc, mesh, &
      problemc_projc)!, symmetric=.true. )
    call create_sysmatrix_data ( sysmatrixc_projc )

  end subroutine define_conformation_problem

end program extrudate_swell2d_c
