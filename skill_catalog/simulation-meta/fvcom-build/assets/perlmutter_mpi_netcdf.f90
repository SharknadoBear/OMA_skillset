program probe_parallel_netcdf
  use mpi
  use netcdf
  implicit none
  integer :: ierr, rank, nranks, aggregate, local, ncid, dimid, varid, value(1)
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nranks, ierr)
  local = rank + 1
  call MPI_Allreduce(local, aggregate, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, ierr)
  if (nranks /= 2 .or. aggregate /= 3) call MPI_Abort(MPI_COMM_WORLD, 11, ierr)
  if (rank == 0) then
    call check(nf90_create('cpu_probe.nc', nf90_clobber, ncid))
    call check(nf90_def_dim(ncid, 'samples', 1, dimid))
    call check(nf90_def_var(ncid, 'rank_sum', nf90_int, [dimid], varid))
    call check(nf90_enddef(ncid))
    call check(nf90_put_var(ncid, varid, [aggregate]))
    call check(nf90_close(ncid))
  endif
  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  call check(nf90_open('cpu_probe.nc', nf90_nowrite, ncid))
  call check(nf90_inq_varid(ncid, 'rank_sum', varid))
  call check(nf90_get_var(ncid, varid, value))
  call check(nf90_close(ncid))
  if (value(1) /= 3) call MPI_Abort(MPI_COMM_WORLD, 12, ierr)
  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  if (rank == 0) print '(A)', 'FVCOM_MPI_NETCDF_PASS ranks=2 value=3'
  call MPI_Finalize(ierr)
contains
  subroutine check(status)
    integer, intent(in) :: status
    if (status /= nf90_noerr) then
      print *, rank, trim(nf90_strerror(status))
      call MPI_Abort(MPI_COMM_WORLD, 13, ierr)
    endif
  end subroutine
end program
