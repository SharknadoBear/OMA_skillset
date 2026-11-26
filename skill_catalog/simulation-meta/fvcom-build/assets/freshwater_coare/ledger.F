# if defined (LAKE_COARE_DIAG)
! Each local primary node contributes 1/BN_MLT of its native dual area.
subroutine coare_area_weights(area)
  use all_vars, only: m,dp,art1
  use mod_par, only: nbn,bn_loc,bn_mlt
  use mod_wd, only: iswetn
  implicit none
  real(dp),intent(out)::area(m)
  integer::j
  area=real(art1(1:m),dp)*real(iswetn(1:m),dp)
  do j=1,nbn
    area(bn_loc(j))=area(bn_loc(j))/real(bn_mlt(j),dp)
  end do
end subroutine

subroutine coare_heat_ledger(wfs,radbed)
  use all_vars, only: m,dp,dti,iint,hsens_watts,hlat_watts,lwrad_watts,swrad_watts,wtsurf_watts, &
                       coare_raw_rain,coare_input_sst,coare_signed_vapor,qprec,qevap
  use lims, only: myid
  implicit none
  real(dp),intent(in)::wfs(m),radbed(m)
  real(dp)::a(m),h
  real(dp),save::tot(15)=0.0_dp
  integer,save::unit=0,ticks=0
  character(80)::filename
  if(ticks==0)then
    write(filename,'("coare_heat_rank_",i5.5,".dat")')myid
    open(newunit=unit,file=trim(filename),status="new",action="write")
  end if
  ticks=ticks+1
  call coare_area_weights(a)
  h=real(dti,dp)
  tot(1)=tot(1)+h*sum(a)
  tot(2)=tot(2)+h*sum(a*hsens_watts(1:m))
  tot(3)=tot(3)+h*sum(a*hlat_watts(1:m))
  tot(4)=tot(4)+h*sum(a*lwrad_watts(1:m))
  tot(5)=tot(5)+h*sum(a*swrad_watts(1:m))
  tot(6)=tot(6)+h*sum(a*wtsurf_watts(1:m))
  tot(7)=tot(7)+h*sum(a*(-4186000._dp*wfs))
  tot(8)=tot(8)+h*sum(a*(-4186000._dp*radbed))
  tot(9)=tot(9)+h*sum(a*coare_raw_rain(1:m))
  tot(10)=tot(10)+h*sum(a*coare_signed_vapor(1:m))
  tot(11)=tot(11)+h*sum(a*qprec(1:m))
  tot(12)=tot(12)+h*sum(a*qevap(1:m))
  tot(13)=tot(13)+h*sum(a*1000._dp*(2.501_dp-.00237_dp*coare_input_sst(1:m))*1.e6_dp*coare_signed_vapor(1:m))
  tot(14)=tot(14)+h*sum(a*max(coare_signed_vapor(1:m),0._dp))
  tot(15)=tot(15)+h*sum(a*4186000._dp*coare_input_sst(1:m)*(qprec(1:m)+qevap(1:m)))
  if(mod(ticks,nint(3600._dp/h))==0)then
    write(unit,'(2i12,15es26.16)')ticks,iint,tot
    flush(unit)
  end if
end subroutine

subroutine coare_external_ledger(rain,vapor)
  use all_vars, only: m,mt,dp,sp,dte,iint
  use lims, only: myid
  implicit none
  real(sp),intent(in)::rain(0:mt),vapor(0:mt)
  real(dp)::a(m),h
  real(dp),save::tot(5)=0.0_dp
  integer,save::unit=0,ticks=0
  character(80)::filename
  if(ticks==0)then
    write(filename,'("coare_ext_rank_",i5.5,".dat")')myid
    open(newunit=unit,file=trim(filename),status="new",action="write")
  end if
  call coare_area_weights(a)
  h=real(dte,dp);ticks=ticks+1
  tot(1)=tot(1)+h*sum(a)
  tot(2)=tot(2)+h*sum(a*rain(1:m))
  tot(3)=tot(3)+h*sum(a*vapor(1:m))
  tot(4)=tot(4)+h*sum(a*max(vapor(1:m),0._dp))
  tot(5)=tot(5)+h*sum(a*min(vapor(1:m),0._dp))
  if(mod(ticks,nint(3600._dp/h))==0)then
    write(unit,'(2i12,5es26.16)')ticks,iint,tot
    flush(unit)
  end if
end subroutine
# endif
