! Optional, observational salinity diagnostics. Only local primary nodes contribute.
subroutine estuary_salt_stage(stage,field,depth,wet)
  use all_vars, only: m,mt,kb,kbm1,dz,art1,sp,dp,dti,iint
  use mod_par, only: bn_loc,bn_mlt,nbn
  use lims, only: myid
  implicit none
  integer,intent(in)::stage,wet(0:mt)
  real(sp),intent(in)::field(0:mt,kb),depth(0:mt)
  integer::i,k
  integer,save::unit=0,ticks=0
  real(dp),allocatable,save::weight(:)
  real(dp),save::sums(8)=0.0_dp,previous=0.0_dp,first=0.0_dp,vol0=0.0_dp
  real(dp)::salt,volume,v
  character(70)::filename
  if(.not.allocated(weight))then
    allocate(weight(m));weight=1.0_dp
    do i=1,nbn
      if(bn_loc(i)<=m)weight(bn_loc(i))=1.0_dp/bn_mlt(i)
    end do
    write(filename,'("salt_stage_rank_",i5.5,".dat")')myid
    open(newunit=unit,file=trim(filename),status="new",action="write")
  end if
  salt=0.0_dp;volume=0.0_dp
  do i=1,m
    if(wet(i)/=1)cycle
    v=weight(i)*art1(i)*depth(i)
    volume=volume+v
    do k=1,kbm1
      salt=salt+v*dz(i,k)*field(i,k)
    end do
  end do
  if(stage==1)then
    if(ticks==0)then
      previous=salt;first=salt;vol0=volume
    endif
    ticks=ticks+1
  endif
  sums(stage)=sums(stage)+salt-previous
  previous=salt
  if(stage==8 .and. mod(ticks,nint(360.0_dp/dti))==0)then
    write(unit,'(2i12,12es26.16)')ticks,iint,vol0,volume,first,salt,sums
    flush(unit)
  endif
end subroutine

subroutine estuary_salt_boundary(stage,field)
  use all_vars, only: m,mt,kb,kbm1,dz,art1,sp,dp,dti,iint,d,dtfa,s1
  use mod_par, only: bn_loc,bn_mlt,nbn
  use mod_obcs, only: uard_obcn,salt_obc
  use all_vars, only: qdis,sdis,inodeq,numqbc
  use mod_wd, only: iswetn
  use lims, only: myid
  use bcs, only: i_obc_n,iobcn
  implicit none
  real(sp),intent(in)::field(0:mt,kb)
  integer,intent(in)::stage
  integer::i,j,k
  integer,save::unit=0,ticks=0
  real(dp),allocatable,save::weight(:),prior(:,:)
  real(dp),save::sums(11)=0.0_dp,maxerr=0.0_dp
  real(dp)::w,q,delta,source,inside,observed
  character(70)::filename
  if(.not.allocated(weight))then
    allocate(weight(m),prior(iobcn,kbm1));weight=1.0_dp
    do i=1,nbn
      if(bn_loc(i)<=m)weight(bn_loc(i))=1.0_dp/bn_mlt(i)
    enddo
    write(filename,'("salt_boundary_rank_",i5.5,".dat")')myid
    open(newunit=unit,file=trim(filename),status="new",action="write")
  endif
  if(stage==1)then
    do i=1,iobcn
      prior(i,:)=field(i_obc_n(i),1:kbm1)
    enddo
    return
  endif
  ticks=ticks+1
  do i=1,numqbc
    j=inodeq(i)
    if(j<1 .or. j>m)cycle
    sums(10)=sums(10)+weight(j)*dti*qdis(i)
    sums(11)=sums(11)+weight(j)*dti*qdis(i)*sdis(i)
  enddo
  do i=1,iobcn
    j=i_obc_n(i)
    if(j>m)cycle
    if(iswetn(j)/=1)cycle
    w=weight(j);q=uard_obcn(i);delta=0.0_dp;source=0.0_dp;inside=0.0_dp;observed=0.0_dp
    do k=1,kbm1
      delta=delta+w*art1(j)*dtfa(j)*dz(j,k)*(field(j,k)-prior(i,k))
      source=source+dz(j,k)*salt_obc(i,k)
      inside=inside+dz(j,k)*s1(j,k)
      observed=observed+dz(j,k)*field(j,k)
      if(q<=0.0_dp)maxerr=max(maxerr,abs(field(j,k)-salt_obc(i,k)))
    enddo
    if(q>0.0_dp)then
      sums(1)=sums(1)+q*dti*w
      sums(3)=sums(3)+q*dti*w*inside
      sums(5)=sums(5)+delta
      sums(7)=sums(7)+w
    else
      sums(2)=sums(2)-q*dti*w
      sums(4)=sums(4)-q*dti*w*source
      sums(6)=sums(6)+delta
      sums(8)=sums(8)+w
      sums(9)=sums(9)-q*dti*w*observed
    endif
  enddo
  if(mod(ticks,nint(360.0_dp/dti))==0)then
    write(unit,'(2i12,12es26.16)')ticks,iint,sums,maxerr
    flush(unit)
  endif
end subroutine
