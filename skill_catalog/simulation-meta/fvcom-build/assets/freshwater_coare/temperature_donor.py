"""A separately built, source-pinned first-order donor-face experiment for native temperature."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

PIN="1835493c31c9bea1e2022d41efac5dd7916e0d1386d7208469b4b497beeeefe9"
FACES="""        DONOR_HEAT_FACE(1)=0.0_SP
        DONOR_HEAT_FACE(KB)=0.0_SP
        DO K=2,KBM1
           IF(WTS(I,K)>=0.0_SP)THEN
              DONOR_HEAT_FACE(K)=WTS(I,K)*T1(I,K)
           ELSE
              DONOR_HEAT_FACE(K)=WTS(I,K)*T1(I,K-1)
           END IF
        END DO"""
NET="             TEMP=DONOR_HEAT_FACE(K)-DONOR_HEAT_FACE(K+1)"
def replace_once(source, anchor, replacement):
    if source.count(anchor) != 1:
        raise ValueError("Expected a unique native source anchor: " + anchor[:90])
    return source.replace(anchor, replacement)


def patch(root):
    path = root / "adv_t.F"
    original = path.read_bytes()
    if hashlib.sha256(original).hexdigest() != PIN:
        raise ValueError("Expected the frozen GNU signed freshwater adv_t.F")
    declaration = "  REAL(SP), DIMENSION(0:MT,KB)      :: XFLUX,XFLUX_ADV,RF"
    source = replace_once(original.decode(), declaration, declaration+"\n  REAL(SP), DIMENSION(KB) :: DONOR_HEAT_FACE")

    section = "  !   The central difference scheme in vertical advection"
    preceding, vertical = source.split(section)
    face_anchor = "#       if defined (NH) || defined (LIMITER_VER_ADV)\n        T_TEMP(0)  = -T1(I,1)"
    fill_faces = ("#       if defined(FRESHWATER_LAKE) && !defined(NH) && !defined(LIMITER_VER_ADV)\n"
                  "        IF(.NOT. BACKWARD_ADVECTION)THEN\n" + FACES + "\n        END IF\n#       endif\n")
    vertical = replace_once(vertical, face_anchor, fill_faces + face_anchor)
    forward = "           IF(.NOT. BACKWARD_ADVECTION)THEN\n"
    start = vertical.index(forward + "             IF(K == 1) THEN") + len(forward)
    end = vertical.index("           ELSE\n             IF(BACKWARD_STEP==1)THEN", start)
    vertical = (vertical[:start] + "#if defined(FRESHWATER_LAKE)\n" + NET + "\n#else\n"
                + vertical[start:end] + "#endif\n" + vertical[end:])
    transformed = (preceding + section + vertical).encode()
    path.write_bytes(transformed)
    return dict(source_file=path.name, before=PIN, after=hashlib.sha256(transformed).hexdigest(),
                field="T1 only", positive_wts_donor="lower", negative_wts_donor="upper", upper_and_bed_faces="zero")


PROBE_PROGRAM="""
program probe
  implicit none
  integer,parameter::sp=selected_real_kind(15),kb=7,kbm1=kb-1
  integer::i,k,j,profile,code,test,n,total
  real(sp)::donor_heat_face(kb),wts(1,kb),temp,t1(1,kb),thick(kbm1),next(kbm1),mass(kbm1),new(kbm1),&
            c,err,worst,cold,hot,initial_total
  i=1; worst=0; cold=0; hot=0; total=0
  thick=(/1.0_sp,1.4_sp,0.7_sp,2.1_sp,3.0_sp,0.8_sp/)
  do profile=0,728
    n=profile
    do k=1,kbm1
      t1(i,k)=mod(n,3);n=n/3
    end do
    do code=0,242
      n=code
      wts=0
      do k=2,kbm1
        wts(i,k)=0.29_sp*real(mod(n,3)-1,sp);n=n/3
      end do
__FACES__
      do k=1,kbm1
__NET__
        next(k)=thick(k)+wts(i,k+1)-wts(i,k)
        if(next(k)<=0)stop 2
        mass(k)=thick(k)*t1(i,k)-temp
        new(k)=mass(k)/next(k)
      end do
      initial_total=sum(thick*t1(i,1:kbm1))
      worst=max(worst,abs(sum(next*new)-initial_total))
      cold=max(cold,minval(t1(i,1:kbm1))-minval(new))
      hot=max(hot,maxval(new)-maxval(t1(i,1:kbm1)))
      total=total+1
    end do
  end do
  write(*,'(i10,3es24.14)')total,worst,cold,hot
  if(worst>1e-12_sp .or. cold>1e-12_sp .or. hot>1e-12_sp)stop 3
end program
"""


def probe(source, evidence):
    native = (source / "adv_t.F").read_text()
    if native.count(FACES) != 1 or native.count(NET) != 1:
        raise ValueError("Native operator differs from the test")
    program = PROBE_PROGRAM.replace("__FACES__",FACES).replace("__NET__",NET)
    test_source, binary = evidence / "donor_probe.f90", evidence / "donor_probe"
    test_source.write_text(program)
    subprocess.run(["ftn","-O2",str(test_source),"-o",str(binary)],check=True)
    cases, conservation, lower, upper = subprocess.check_output([str(binary)],text=True).split()
    return dict(native_cases=int(cases), max_extensive_error=float(conservation), lower_range_error=float(lower),
                upper_range_error=float(upper), layers=6, face_velocity_patterns=243, temperature_profiles=729)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("evidence",type=Path)
    args = parser.parse_args()
    (args.evidence/"donor_source.json").write_text(json.dumps(patch(args.source),indent=2)+"\n")
    (args.evidence/"donor_operator.json").write_text(json.dumps(probe(args.source,args.evidence),indent=2)+"\n")
