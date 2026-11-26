"""Source-locked Lake Superior COARE26Z diagnostic. All added paths require LAKE_COARE_DIAG."""
from pathlib import Path
import hashlib,json,sys
BASE={"mod_force.F":"4ce9f2c29423fdabf39808eb62b966b92e8b0500a6c3e8b4dedd8c1c26c840ba",
      "bcond_gcn.F":"861a0c4b50c6234d93ee56a9c21a98900e67c73517a78095ad68d8ee7cf29559",
      "mod_main.F":"f84d643cef9dcd0ce0f909633acc054cac1d22aeb02cfe284a3234e8881d759f",
      "mod_ncdio.F":"47771e724b631b8e44ca9a44845f4295d8e6df7bbab42be4f3c30f8e3982cc00",
      "vdif_ts.F":"7457c52cff30d1d285f0e2c13543514bf7dcd45a9cd757b69fda5ef38197f3ee"}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sub(t,b,a):
    if t.count(b)!=1:raise ValueError("nonunique or missing source: "+repr(b[:90])+" count="+str(t.count(b)))
    return t.replace(b,a)
def guarded(t):return "# if defined (LAKE_COARE_DIAG)\n"+t.strip("\n")+"\n# endif\n"
def force(t):
    start=t.index("  SUBROUTINE SURFACE_HEATING_CALCULATED\n")
    end=t.index("  END SUBROUTINE SURFACE_HEATING_CALCULATED",start)
    t0,t1,t2=t[:start],t[start:end],t[end:]
    marker="  TYPE(NCVAR), POINTER :: T_AIR_N, T_AIR_P"
    at=t0.index(marker)
    t0=t0[:at]+guarded("  TYPE(NCVAR), POINTER :: BULK_U_N,BULK_U_P,BULK_V_N,BULK_V_P")+t0[at:]
    marker='    CASE(HEAT_IS_FVCOMGRID)'
    left,right=t1.split(marker)
    key="       ! SETUP THE ACTUAL VARIABLES USED TO LOAD DATA!"
    code=[]
    for axis in ("U","V"):
        code.append(f'       VAR => FIND_VAR(HEAT_FILE,"{axis}10",FOUND)')
        code.append(f'       IF (.NOT.FOUND) CALL FATAL_ERROR("Lake bulk file requires {axis}10 on nele")')
        for side in ("P","N"):
            sym="BULK_"+axis+"_"+side
            code.extend([f"       {sym} => REFERENCE_VAR(VAR)","       ALLOCATE(STORAGE_VEC(0:NT),STAT=STATUS)",
                '       IF(STATUS/=0) CALL FATAL_ERROR("Unable to allocate lake bulk-vector bracket")',
                f"       CALL NC_CONNECT_PVAR({sym},STORAGE_VEC)","       NULLIFY(STORAGE_VEC)",f"       {sym}%CURR_STKCNT=0"])
    right=sub(right,key,guarded("\n".join(code))+key)
    t1=left+marker+right
    t=t0+t1+t2
    begin=t.index("  SUBROUTINE UPDATE_HEAT_CALCULATED(");end=t.index("  END SUBROUTINE UPDATE_HEAT_CALCULATED",begin)
    left,body,right=t[:begin],t[begin:end],t[end:]
    key="    REAL(SP), DIMENSION(0:NT) :: WDSN"
    body=sub(body,key,guarded("""    REAL(SP), DIMENSION(0:NT) :: HEAT_U10,HEAT_V10
    INTEGER,SAVE :: TRACE_CALLS=0,TRACE_UNIT=0
    CHARACTER(64) :: TRACE_PATH""")+key)
    key='    IF(WIND_TYPE /= \'speed\')CALL FATAL_ERROR("WIND_TYPE must be \'speed\' for heat flux calculating")'
    gate="""    IF(WIND_TYPE/=STRESS .OR. HEATING_CALCULATE_KIND/=VRBL .OR. HEAT_FORCING_TYPE/=HEAT_IS_FVCOMGRID) &
       CALL FATAL_ERROR("Lake COARE diagnostic requires prescribed stress and native variable heat")
    IF(.NOT.HEATING_FRESHWATER .OR. TRIM(COARE_VERSION)/='COARE26Z') &
       CALL FATAL_ERROR("Lake COARE diagnostic requires freshwater COARE26Z")
    IF(ZUU/=10._SP .OR. ZTT/=2._SP .OR. ZQQ/=2._SP) CALL FATAL_ERROR("Unexpected COARE observation heights")"""
    body=sub(body,key,"# if defined (LAKE_COARE_DIAG)\n"+gate+"\n# else\n"+key+"\n# endif")
    marker="    CASE(HEAT_IS_FVCOMGRID)"
    head,tail=body.split(marker)
    key="       WDSN(1:NT) = SQRT(UUWIND(1:NT)*UUWIND(1:NT)+VVWIND(1:NT)*VVWIND(1:NT))"
    code=[]
    for axis in ("U","V"):
        prefix="BULK_"+axis
        code.extend([f"       CALL UPDATE_VAR_BRACKET(HEAT_FILE,{prefix}_P,{prefix}_N,HTIME,STATUS)",
                     f'       IF(STATUS/=0) CALL FATAL_ERROR("Lake COARE {axis}10 outside bracket")',
                     f"       CALL NC_POINT_VAR({prefix}_N,VNP)",f"       CALL NC_POINT_VAR({prefix}_P,VPP)",
                     f"       HEAT_{axis}10 = FTM%NEXT_WGHT*VNP + FTM%PREV_WGHT*VPP"])
    code.append("       WDSN(1:NT)=SQRT(HEAT_U10(1:NT)**2+HEAT_V10(1:NT)**2)")
    code.append("       TRACE_CALLS=TRACE_CALLS+1")
    tail=sub(tail,key,"# if defined (LAKE_COARE_DIAG)\n"+"\n".join(code)+"\n# else\n"+key+"\n# endif")
    key="         WDS = WDS/FLOAT(NTVE(I))"
    tail=sub(tail,key,key+"\n"+guarded("         COARE_WIND_SPEED(I)=WDS"))
    end_loop=tail.rindex("       END DO")
    tail=tail[:end_loop]+guarded("""         IF(TRACE_CALLS==1 .OR. MOD(HTIME%MuSOD,1800000000_8)==0_8)THEN
           IF(ANY(NGID(I)==(/694,39652,69552/)))THEN
             IF(TRACE_UNIT==0)THEN
               WRITE(TRACE_PATH,'("coare_point_rank_",I5.5,".dat")')MYID
               OPEN(NEWUNIT=TRACE_UNIT,FILE=TRIM(TRACE_PATH),STATUS="new",ACTION="write")
             END IF
             WRITE(TRACE_UNIT,'(4I17,14ES26.16)')IINT,HTIME%MJD,HTIME%MuSOD,NGID(I), &
                 T1(I,1),COARE_WIND_SPEED(I),T_AIR(I),RH_AIR(I),PA_AIR(I),DLW_AIR(I),DSW_AIR(I), &
                 CORR(I),HEAT_SENSIBLE(I),HEAT_LATENT(I),HEAT_RLN(I),HEAT_NET(I),USR,DTER
             FLUSH(TRACE_UNIT)
           END IF
         END IF""")+tail[end_loop:]
    return left+head+marker+tail+right
def boundary(t):
    exchange="         IF(PAR)CALL AEXCHANGE(NC,MYID,NPROCS,HSENS_WATTS,HLAT_WATTS,LWRAD_WATTS)"
    begin=t.index("         CALL UPDATE_HEAT_CALCULATED(")
    end=t.index("# endif",t.index(exchange,begin))
    t=t[:end]+guarded("         IF(PAR)CALL AEXCHANGE(NC,MYID,NPROCS,COARE_WIND_SPEED)")+t[end:]
    key="   INTEGER, INTENT(IN) :: IDX"
    t=sub(t,key,guarded("   REAL(SP), ALLOCATABLE, SAVE :: LAKE_VAPOR(:)")+key)
    key="         CALL UPDATE_PRECIPITATION(IntTime,Qprec,Qevap)"
    t=sub(t,key,key+"\n"+guarded("""         IF (.NOT.ALLOCATED(LAKE_VAPOR)) ALLOCATE(LAKE_VAPOR(0:MT))
         IF (.NOT.HEATING_CALCULATE_ON .OR. ANY(Qevap(1:M)/=0._SP)) &
             CALL FATAL_ERROR("Lake COARE needs calculated heat and raw rain only")
         ! Same latent heat and surface temperature used by the immediately preceding COARE evaluation.
         COARE_INPUT_SST = T1(:,1)
         COARE_RAW_RAIN = Qprec
         LAKE_VAPOR = HLAT_WATTS / (1000._SP*(2.501_SP-0.00237_SP*T1(:,1))*1.0E6_SP)
         COARE_SIGNED_VAPOR = LAKE_VAPOR
         Qprec = Qprec + MAX(LAKE_VAPOR,0._SP)
         Qevap = MIN(LAKE_VAPOR,0._SP)"""))
    key="         CALL UPDATE_PRECIPITATION(ExtTime,Qprec2,Qevap2)"
    t=sub(t,key,key+"\n"+guarded("""         IF (.NOT.ALLOCATED(LAKE_VAPOR) .OR. ANY(Qevap2(1:M)/=0._SP)) &
             CALL FATAL_ERROR("Lake COARE split before validated internal vapor/dew")
         ! Use the same internal-interval latent energy in both 2.5 s external steps.
         ! Raw rain is still reconstructed at each external clock separately.
         CALL COARE_EXTERNAL_LEDGER(Qprec2,LAKE_VAPOR)
         Qprec2 = Qprec2 + MAX(LAKE_VAPOR,0._SP)
         Qevap2 = MIN(LAKE_VAPOR,0._SP)"""))
    return t
DIAG=(("COARE_INPUT_SST","degrees_C"),("COARE_RAW_RAIN","m s-1"),("COARE_SIGNED_VAPOR","m s-1"),("COARE_WIND_SPEED","m s-1"))
def main_vars(t):
    key="   ! SURFACE MET STUFF"
    t=sub(t,key,key+"\n"+guarded("\n".join("   REAL(SP), ALLOCATABLE,TARGET :: "+k+"(:)" for k,_ in DIAG)))
    key="   ALLOCATE(WTSURF(0:MT))"
    return sub(t,key,guarded("\n".join("   ALLOCATE("+k+"(0:MT)); "+k+"=ZERO" for k,_ in DIAG))+key)
def io(t):
    i=t.index("  FUNCTION SURFACE_HEATING_FILE_OBJECT()");j=t.index("END FUNCTION SURFACE_HEATING_FILE_OBJECT",i)
    body=t[i:j];key="       allocate(SWRAD_WATTS(MGL),stat=status)"
    body=sub(body,key,guarded("\n".join("       ALLOCATE("+k+"(MGL)); "+k+"=0._SP" for k,_ in DIAG))+key)
    i0=body.index("    VAR  => NC_MAKE_AVAR(name='net_heat_flux'");i1=body.index("    NCF  => ADD(NCF,VAR)",i0)+len("    NCF  => ADD(NCF,VAR)")
    block=body[i0:i1];extra=[]
    for name,units in DIAG:
        extra.append(block.replace("net_heat_flux",name.lower()).replace("WTSURF_WATTS",name).replace("Surface Net Heat Flux",name.lower().replace("_"," ")).replace("W m-2",units))
    return t[:i]+body[:i1]+"\n"+guarded("\n".join(extra))+body[i1:]+t[j:]
def vdif(t):
    mark="  CASE(2)\n\n     SWRADF"
    t=sub(t,mark,guarded("     CALL COARE_HEAT_LEDGER(WFSURF,RAD(:,KB))")+"\n"+mark)
    return t+"\n"+Path(__file__).with_name("ledger.F").read_text()+"\n"
def run(root,receipt):
    outputs={}
    for filename,transform in (("mod_force.F",force),("bcond_gcn.F",boundary),("mod_main.F",main_vars),("mod_ncdio.F",io),("vdif_ts.F",vdif)):
        p=root/filename
        if sha(p)!=BASE[filename]:raise ValueError("unrecognized frozen source "+filename)
        old=p.read_bytes().decode("latin1")
        new=transform(old)
        p.write_bytes(new.encode("latin1"));outputs[filename]=dict(before=BASE[filename],after=sha(p))
    receipt.write_text(json.dumps(outputs,indent=2)+"\n")
if __name__=="__main__":run(Path(sys.argv[1]),Path(sys.argv[2]))
