#!/bin/bash
# Runs within a Perlmutter CPU allocation. The candidate subprocess owns its module environment.
set -euo pipefail
payload=$(cd -- "$(dirname -- "$0")" && pwd -P)
build=$(cd -- "${SLURM_SUBMIT_DIR:?submit from the study run/build directory}" && pwd -P)
case "$build" in */run/build) ;; *) echo "Submit from STUDY/run/build" >&2; exit 2;; esac
root=$(cd -- "$build/../.." && pwd -P)
job=${SLURM_JOB_ID:?Slurm allocation required}
attempt="$build/attempt_$job"
source_root="$root/source"
if [[ ${1:-} != --candidate ]]; then
  [[ ${NERSC_HOST:-} == perlmutter ]] || { echo "Expected Perlmutter CPU host" >&2; exit 2; }
  mkdir -p "$source_root"
  mkdir "$attempt"
  test ! -e "$source_root/fvcom_plased"
  (cd "$payload"; sha256sum -c payload.sha256) > "$attempt/payload_check.log"
  for family in gnu cray; do
    work="$attempt/$family"
    mkdir "$work"
    if bash "$payload/perlmutter_build.sh" --candidate "$family" > "$work/driver.log" 2>&1; then
      evidence="$work/evidence"
      test -s "$evidence/freeze.json"
      ln "$source_root/build_${job}_$family/fvcom_plased" "$source_root/fvcom_plased"
      (cd "$source_root"; sha256sum -c "$evidence/executable.sha256")
      tar -C "$work" -czf "$attempt/evidence_$job.tgz" evidence
      (cd "$attempt"; sha256sum "evidence_$job.tgz" > evidence.sha256)
      printf '%s\n' "$family" > "$attempt/selected_environment"
      echo "FVCOM_BUILD_PASS job=$job family=$family"
      exit 0
    fi
    echo "$family candidate failed; recorded in $work" >&2
    tail -n 12 "$work/driver.log" >&2
  done
  exit 1
fi

family=${2:?compiler family}
work="$attempt/$family"
src="$source_root/build_${job}_$family"
evidence="$work/evidence"
mkdir "$src" "$evidence"
cd "$work"
exec > "$evidence/build.log" 2>&1
set -x
set +u
module reset
if module is-loaded gpu; then module swap gpu cpu; else module load cpu; fi
if module is-loaded cudatoolkit; then module unload cudatoolkit; fi
module load "PrgEnv-$family"
module load cray-mpich
# Prefer the jointly installed Cray MPI/HDF5/netCDF Fortran stack.
if module load cray-hdf5-parallel cray-netcdf-hdf5parallel; then
  hdf5=cray-hdf5-parallel
  netcdf=cray-netcdf-hdf5parallel
else
  module unload cray-netcdf-hdf5parallel cray-hdf5-parallel 2>/dev/null || true
  module load cray-hdf5 cray-netcdf
  hdf5=cray-hdf5
  netcdf=cray-netcdf
fi
case ":${LOADEDMODULES:-}:" in
  *:gpu/*:*|*:gpu:*|*:craype-accel-*:*|*:cudatoolkit/*:*) echo "GPU stack active in CPU build" >&2; exit 42;;
esac
export CRAYPE_LINK_TYPE=dynamic
set -u
find_python() {
  local candidate
  for candidate in /usr/bin/python3.11 python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null && "$candidate" -c 'import sys; assert sys.version_info >= (3,10); print(sys.executable)' 2>/dev/null; then
      return 0
    fi
  done
  return 1
}
python_module=
if ! python_exe=$(find_python); then
  python_module=python
  python_exe=$(set +u; module load "$python_module" && find_python)
fi
run_python() (
  if [[ -n $python_module ]]; then set +u; module load "$python_module"; set -u; fi
  "$python_exe" "$@"
)
run_python -c 'import sys; print(sys.executable); print(sys.version)' > "$evidence/python_runtime.txt"
module -t list > "$evidence/modules.txt" 2>&1
hostname > "$evidence/hostname.txt"
cat /etc/os-release > "$evidence/os-release.txt"
printf '%s\n' "$NERSC_HOST" > "$evidence/nersc_host.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$evidence/build_job_id.txt"
printf '%s\n' "${SLURM_JOB_CONSTRAINTS:-}" > "$evidence/slurm_constraints.txt"
scontrol show job -o "$SLURM_JOB_ID" > "$evidence/scheduler_job.txt"
grep -Eq '(^|[[:space:]])Features=([^ ]*[,|&])?cpu([,|&][^ ]*)?([[:space:]]|$)' "$evidence/scheduler_job.txt"
{ command -v ftn; ftn --version; command -v cc; cc --version; } > "$evidence/compiler.txt" 2>&1
{ module show "$netcdf"; if command -v nf-config >/dev/null; then nf-config --all; fi; } > "$evidence/nf-config.txt" 2>&1
cp "$payload/perlmutter_build.sh" "$evidence/build_commands.txt"
cp "$payload/perlmutter_mpi_netcdf.f90" "$evidence/"
ftn -craype-verbose "$payload/perlmutter_mpi_netcdf.f90" -o probe > "$evidence/wrapper_link.txt" 2>&1
export OMP_NUM_THREADS=1
srun --exact --ntasks=2 --cpus-per-task=1 --cpu-bind=cores ./probe > "$evidence/mpi_netcdf_probe.log" 2>&1
grep -Fx 'FVCOM_MPI_NETCDF_PASS ranks=2 value=3' "$evidence/mpi_netcdf_probe.log"
cp cpu_probe.nc "$evidence/"
cp "$payload/"{source_inventory_before.sha256,source_revision.json} "$evidence/"
cd "$src"
tar xzf "$payload/pinned_source.tgz"
sha256sum -c "$evidence/source_inventory_before.sha256" > "$evidence/source_inventory_verification.log"
cp make.inc "$evidence/make.inc.before"
cp mod_plast.F "$evidence/mod_plast.F.before"
patch -p1 --forward < "$payload/no_plastic_guard.patch"
diff -u "$evidence/mod_plast.F.before" mod_plast.F > "$evidence/no_plastic_guard.diff" || test "$?" = 1
cp mod_utils.F "$evidence/mod_utils.F.before"
cp mod_interp.F "$evidence/mod_interp.F.before"
cp "$payload/fvcom_mp_sign_kind.patch" "$evidence/sign_kind.diff"
patch -p1 --fuzz=0 --forward < "$evidence/sign_kind.diff"
sha256sum mod_utils.F mod_interp.F mod_plast.F > "$evidence/patched_sources.sha256"
printf '%s\n' 'Both SIGN operands must use the same Fortran kind. The typed 1.0_sp magnitude is exact in either kind. All sign arguments and expressions are preserved.' > "$evidence/sign_kind_reason.txt"
case "$family" in
  gnu) macro=GFORTRAN; options='-O2 -ffree-line-length-none -fallow-argument-mismatch -fno-range-check'; julian_fc='ftn -fallow-argument-mismatch';;
  cray) macro=CRAY; options='-O2'; julian_fc=ftn;;
  *) exit 3;;
esac
printf 'TOPDIR = %s\nCOMPILER = -D%s\nOPT = %s\n' "$src" "$macro" "$options" > make.inc
cat "$payload/perlmutter_make.inc" >> make.inc
cp make.inc "$evidence/make.inc.after"
diff -u "$evidence/make.inc.before" make.inc > "$evidence/make.inc.diff" || test "$?" = 1
run_python "$payload/fvcom_build.py" audit-make-inc --air-pressure --make-inc make.inc --expected-topdir "$src" > "$evidence/make_inc_audit.json"
cd libs
tar xzf metis.tgz
tar xzf julian.tgz
cp metis/rename.h "$evidence/metis_rename.h.before"
sed -i 's/__log2/__fvcom_metis_log2/g' metis/rename.h
diff -u "$evidence/metis_rename.h.before" metis/rename.h > "$evidence/metis_rename.diff" || test "$?" = 1
mkdir -p install/{include,lib,bin}
cd "$src"
make -C libs/metis -j8 install
make -C libs/julian -j8 libjulian CC='cc -std=gnu89 -Wno-implicit-function-declaration -Wno-incompatible-pointer-types' FC="$julian_fc"
install -m 644 libs/julian/libjulian.a libs/install/lib/
install -m 644 libs/julian/fjulian.inc libs/install/include/
make clobber
make -j8
sha256sum fvcom_plased > "$evidence/executable.sha256"
file fvcom_plased > "$evidence/executable.file.txt"
ldd fvcom_plased > "$evidence/executable.ldd.txt"
! grep -F 'not found' "$evidence/executable.ldd.txt" || exit 41
mkdir -p "$work/reader/"{run,input,output}
cd "$work/reader/run"
srun --exact --ntasks=1 --cpus-per-task=1 --cpu-bind=cores "$src/fvcom_plased" --create_namelist=blank > "$evidence/blank_probe.log" 2>&1
cp blank_run.nml "$evidence/blank_namelist.nml"
cp "$payload/river.nml" ../input/
for member in control combined; do
  cp "$payload/probe_${member}_run.nml" probe_run.nml
  cp probe_run.nml "$evidence/probe_${member}_run.nml"
  reader_status=0
  srun --exact --ntasks=1 --cpus-per-task=1 --cpu-bind=cores "$src/fvcom_plased" --casename=probe --dbg=4 > "$evidence/${member}_reader.log" 2>&1 || reader_status=$?
  grep -Fi 'Subroutine Ends: Read_Name_List;' "$evidence/${member}_reader.log"
  grep -Fi 'expected_read_end.dat NOT FOUND' "$evidence/${member}_reader.log"
  echo "$member reader completed namelist and reached intentional OBC sentinel (rc=$reader_status)." >> "$evidence/smoke.log"
done
echo FVCOM_INPUT_SMOKE_PASS >> "$evidence/smoke.log"
cd "$evidence"
run_python - "$family" "$netcdf" "$hdf5" "$options" "$src" "$job" <<'PY'
import datetime,json,os,pathlib,sys
family,netcdf,hdf5,opts,source_dir,job=sys.argv[1:]
loaded=os.environ['LOADEDMODULES'].split(':')
def module_name(name):
    matches=[x for x in loaded if x==name or x.startswith(name+'/')]
    if len(matches)!=1: raise ValueError('Cannot bind module '+name+': '+repr(matches))
    return matches[0]
env={'site':'perlmutter','os':'cpe','compiler':'ftn','programming_environment':'PrgEnv-'+family,
     'modules':[module_name(x) for x in ('cpu','PrgEnv-'+family,'cray-mpich',hdf5,netcdf)],
     'compiler_options':opts.split()}
revision=json.loads(pathlib.Path('source_revision.json').read_text())
freeze={k:revision[k] for k in ('source_repository','source_commit')}
freeze.update(execution_site='perlmutter',build_environment=env,build_source_dir=source_dir,job_id=job,
              source_edits_after_freeze_allowed=False,executable_sha256=pathlib.Path('executable.sha256').read_text().split()[0],
              frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
pathlib.Path('freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
PY
