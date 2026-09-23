module find_normals_m
   
  use tfem_m

  implicit none

  real(dp), allocatable, dimension(:,:) :: x
  real(dp), allocatable, dimension(:) :: normal

contains 

! subroutine to find normals on a curve in the mesh

subroutine find_normals_curve ( mesh, curve, normals, nx, ny, nz )

    use limits_m

    type(mesh_t), intent(in) :: mesh
    integer, intent(in) :: curve
    type(vector_t), intent(out) :: normals
    type(subscriptvec_t), intent(out) :: nx, ny
    type(subscriptvec_t), intent(out), optional :: nz

    ! temporary mesh for plotting the geometry
    type(mesh_t) :: mesh_plot

    ! problem for normals
    type(input_probdef_t) :: input_probdef
    type(problem_t), target :: problem


    integer :: i
    real(dp), allocatable, dimension(:,:) :: bd
    logical :: warn_old

    call mesh_skeleton ( mesh_plot, mesh%curves(curve)%nnodes, &
      mesh%curves(curve)%nelem, mesh%curves(curve)%element%elshape, &
      mesh%curves(curve)%ndim )

    mesh_plot%coor = mesh%coor(mesh%curves(curve)%nodes,:)
    mesh_plot%topology(1)%a = mesh%curves(curve)%topology(:,:,1)

!   add blocks manually to avoid 'domain of blocks near zero' in fill_mesh_parts

    allocate(bd(mesh%ndim,2))

    do i=1,mesh%ndim
      bd(i,1)=minval(mesh%coor(:,i))-1.e-10_dp
      bd(i,2)=maxval(mesh%coor(:,i))+1.e-10_dp
    end do

    call add_to_mesh ( mesh_plot, blocks=[(2,i=1,mesh%ndim)], &
      blocksdomain=bd )

!   temporarily disable warning for fill_mesh_parts_sidelem (in limits_m)

    warn_old = WARN_ON_TEST_FOR_MULTIPLE_SIDELEM
    WARN_ON_TEST_FOR_MULTIPLE_SIDELEM = .false.

    call fill_mesh_parts ( mesh_plot )

    ! create a problem to write the normals

    call create_input_probdef ( mesh_plot, input_probdef, nvec=1 )

    input_probdef%elementdof(1)%a = 1
    input_probdef%vec_elementdof(1)%a(:,1) = mesh%ndim

    call problem_definition ( input_probdef, mesh_plot, problem )

    call create ( problem, normals, vec=1 )

    call create_subscript_vector ( mesh_plot, problem, nx, degfd=1, vec=1 )
    call create_subscript_vector ( mesh_plot, problem, ny, degfd=2, vec=1 )

    ! approximate the normal in each nodal point
    ! NOTE: strictly speaking, the normal at a nodal point is ill-defined. Here
    ! we approximate the normal for each element and add them to the nodal
    ! nodal points connected to that element

    call derive_vector ( mesh_plot, problem, vector=normals, &
    elemsub=deriv_normals )

    allocate ( normal(mesh_plot%ndim) )

    ! normalize the normal vectors to length 1

    if ( mesh_plot%ndim == 2 ) then

    do i = 1, size(nx%s)
    normal = [ normals%u(nx%s(i)), normals%u(ny%s(i)) ]
    normal = normal / sqrt(dot_product(normal,normal))
    normals%u(nx%s(i)) = normal(1)
    normals%u(ny%s(i)) = normal(2)
    end do

    else

    call create_subscript_vector ( mesh_plot, problem, nz, degfd=3, vec=1 )

    do i = 1, size(nx%s)
    normal = [ normals%u(nx%s(i)), normals%u(ny%s(i)), &
                                                      normals%u(nz%s(i)) ]
    normal = normal / sqrt(dot_product(normal,normal))
    normals%u(nx%s(i)) = normal(1)
    normals%u(ny%s(i)) = normal(2)
    normals%u(nz%s(i)) = normal(3)
    end do

    end if

    deallocate ( normal )
    call delete ( input_probdef )

  end subroutine find_normals_curve

  !   approximate the normal to an interface mesh

    subroutine deriv_normals ( mesh, problem, elgrp, elem, first, last, &
      coefficients, oldvectors, elemvec, elemwts )

      use math_defs_m

      type(mesh_t), intent(in) :: mesh
      type(problem_t), intent(in) :: problem
      integer, intent(in) :: elgrp, elem
      logical, intent(in) :: first, last
      type(coefficients_t), intent(in) :: coefficients
      type(oldvectors_t), intent(in) :: oldvectors
      real(dp), intent(out), dimension(:) :: elemvec, elemwts

      integer :: numnod, pt(3), ndim

      numnod = mesh%element(1)%numnod
      ndim = mesh%ndim

      if ( first ) allocate ( x(numnod,ndim), normal(ndim) )

!     get coordinates of nodes of interface

      call get_coordinates ( mesh, elgrp, elem, x )

      if ( ndim == 2 ) then

!       approximate the normal on the line element

        select case ( mesh%element(1)%elshape )
          case (1)
            normal = [(x(2,2)-x(1,2)), -(x(2,1)-x(1,1))]
          case (2)
            normal = [(x(3,2)-x(1,2)), -(x(3,1)-x(1,1))]
          case default
            write(*,'(/a/a,i0/)')'deriv_normals: ', &
                ' this elementshape is not available: ', mesh%element(1)%elshape
            stop
        end select

      else

!       get the points to determine the normal on the surface element

        select case ( mesh%element(1)%elshape )
          case (3,10)
            pt = [1,2,3]
          case (4,7)
            pt = [1,3,5]
          case (5,9)
            pt = [1,2,4]
          case (6)
            pt = [1,3,7]
          case default
            write(*,'(/a/a,i0/)')'deriv_normals: ', &
                ' this elementshape is not available: ', mesh%element(1)%elshape
            stop
        end select

!       approximate the normal on the surface element using a cross product

        normal = cross_product ( x(pt(2),:)-x(pt(1),:), x(pt(3),:)-x(pt(1),:) )

      end if

!     add the normal to the element vector

      if ( ndim == 2 ) then
        elemvec(:size(elemvec)/2) = normal(1)
        elemvec(size(elemvec)/2+1:) = normal(2)
      else
        elemvec(:size(elemvec)/3) = normal(1)
        elemvec(size(elemvec)/3+1:2*size(elemvec)/3) = normal(2)
        elemvec(2*size(elemvec)/3+1:) = normal(3)
      end if

      elemwts = 1

      if ( last ) deallocate ( x, normal )

    end subroutine deriv_normals

end module find_normals_m