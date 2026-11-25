program verify_lake_physics
  use mod_prec, only: sp, signed_river_tracer
  use eqs_of_state, only: fofonoff_millard
  implicit none
  real(sp) :: t(11), salt(11), p(11), anomaly(11), rho(11)
  real(sp), parameter :: split(2)=[0.3_sp,0.7_sp], layers(2)=[0.25_sp,0.75_sp]
  real(sp) :: local(2,2), prescribed, flux, q, expected, volume, residual
  integer :: it, side, k, kind
  t=[0.0_sp,1.0_sp,2.0_sp,3.0_sp,3.98_sp,4.0_sp,5.0_sp,6.0_sp,8.0_sp,10.0_sp,15.0_sp]
  salt=0.0_sp
  do k=1,3
    p=real((k-1)**2*75,sp)
    call fofonoff_millard(salt,t,p,0.0_sp,anomaly)
    rho=1000.0_sp*(1.0_sp+anomaly)
    do it=1,size(t)
      write(*,'(a,3(1x,f14.7))') 'EOS',p(it),t(it),rho(it)
    end do
    if (k==1) then
      if (any(abs(rho([1,6,10])-[999.842594_sp,999.974958_sp,999.702082_sp])>0.002_sp)) stop 21
      if (rho(5)<=rho(4) .or. rho(5)<=rho(7) .or. abs(rho(5)-maxval(rho))>0.0001_sp) stop 22
    endif
  enddo
  ! Independent temperatures and synthetic salinities at both ends and both layers.
  do kind=1,2
    local=reshape([2.0_sp,4.0_sp,10.0_sp,13.0_sp],[2,2])/real(kind,sp)
    prescribed=20.0_sp/real(kind,sp)
    do it=-1,1
      q=100.0_sp*real(it,sp)
      flux=0.0_sp
      expected=0.0_sp
      volume=0.0_sp
      do side=1,2
        do k=1,2
          volume=volume+q*split(side)*layers(k)
          flux=flux+q*split(side)*layers(k)*signed_river_tracer(q,prescribed,local(side,k))
          if (q<0) then
            expected=expected+q*split(side)*layers(k)*local(side,k)
          else
            expected=expected+q*split(side)*layers(k)*prescribed
          endif
        enddo
      enddo
      residual=abs(expected-flux)
      write(*,'(a,1x,i1,4(1x,f14.7))') 'SIGNED_EDGE',kind,q,volume,flux,residual
      if (abs(volume-q)>1e-10_sp.or.residual>1e-10_sp) stop 23
    enddo
  enddo
  print '(a)', 'FVCOM_LAKE_EOS_SIGNED_EDGE_PASS'
end program
