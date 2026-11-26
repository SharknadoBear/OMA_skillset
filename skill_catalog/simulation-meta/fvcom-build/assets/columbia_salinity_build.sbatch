#!/bin/bash -l
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=00:10:00
#SBATCH --job-name=estuary_salt_build
#SBATCH --output=build_%j.out
#SBATCH --error=build_%j.err
set -eo pipefail
module reset
if module is-loaded gpu; then module swap gpu cpu; else module load cpu; fi
if module is-loaded cudatoolkit; then module unload cudatoolkit; fi
module load PrgEnv-gnu/8.7.0 cray-mpich/9.1.0 cray-hdf5-parallel/1.14.3.9 cray-netcdf-hdf5parallel/4.9.2.3
set -u
export CRAYPE_LINK_TYPE=dynamic OMP_NUM_THREADS=1
payload=$(cd "$SLURM_SUBMIT_DIR" && pwd -P)
cd "$payload"
sha256sum -c --quiet payload.sha256
src=$(cat parent_source.txt)
parent=$(cat parent_evidence.txt)
test -f "$parent/active_dependencies.mk"
cmp "$src/make.inc" "$parent/make.inc.after"
(cd "$src"; sha256sum -c --quiet "$parent/patched_sources.sha256"; sha256sum -c "$parent/executable.sha256")
work="$payload/job_$SLURM_JOB_ID"
mkdir "$work" "$work/source" "$work/evidence" "$work/template"
e="$work/evidence"
cp parent_binding.json "$parent/make_inc_audit.json" "$parent/mpi_netcdf_probe.log" "$e/"
sha256sum "$src/fvcom_plased" > "$e/parent_executable.sha256"
module -t list > "$e/modules.txt" 2>&1
tar -C "$src" --exclude='./.git' -cf - . | tar -C "$work/source" -xf -
cd "$work/source"
for f in make.inc internal_step.F libfvcom.a fvcom_plased; do test ! -L "$f"; done
/usr/bin/python3.11 - "$src" "$work/source" <<'PY'
import pathlib,sys
p=pathlib.Path("make.inc")
s=p.read_text()
old="TOPDIR = "+sys.argv[1]+"\n"
assert s.count(old)==1
p.write_text(s.replace(old,"TOPDIR = "+sys.argv[2]+"\n"))
PY
/usr/bin/python3.11 "$payload/columbia_salinity_build.py" "$work/source" "$payload/columbia_salinity_ledger.F" "$e/instrument.json"
make -k -f makefile -f "$parent/active_dependencies.mk" -j8 fvcom_plased > "$e/build.log" 2>&1
exe="$work/source/fvcom_plased"
test "$(sha256sum "$exe" | cut -d' ' -f1)" != "$(cut -d' ' -f1 "$e/parent_executable.sha256")"
sha256sum "$exe" > "$e/executable.sha256"
ldd "$exe" > "$e/ldd.txt"
if grep -F 'not found' "$e/ldd.txt"; then exit 53; fi
(cd "$work/template"; srun --exact --ntasks=1 --cpus-per-task=1 --cpu-bind=cores "$exe" --create_namelist=blank > "$e/blank.log" 2>&1)
cp "$work/template/blank_run.nml" "$e/blank_namelist.nml"
cd "$e"
/usr/bin/python3.11 - "$exe" "$SLURM_JOB_ID" <<'PY'
import hashlib,json,pathlib,sys
p=pathlib.Path
sha=lambda file:hashlib.sha256(p(file).read_bytes()).hexdigest()
b=json.loads(p("parent_binding.json").read_text())
assert p("parent_executable.sha256").read_text().split()[0]==b["executable_sha256"]
assert set(b["feature_defines"])==set("AIR_PRESSURE DOUBLE_PRECISION GCN GFORTRAN LIMITED_NO MULTIPROCESSOR SINGLE_OUTPUT WET_DRY".split())
assert sha("blank_namelist.nml")==b["blank_template_sha256"]
v={k:b[k] for k in ("build_environment","source_commit","execution_site","feature_defines","mpi_netcdf_probe_sha256")}
v.update(schema="fvcom_estuary_salinity_build_v1", status="frozen", job_id=sys.argv[2], native_executable=sys.argv[1],executable_sha256=sha(sys.argv[1]),parent_executable_sha256=b["executable_sha256"],instrument_sha256=sha("instrument.json"),ledger_sha256=json.loads(p("instrument.json").read_text())["ledger_sha256"],blank_template_sha256=sha("blank_namelist.nml"))
p("freeze.json").write_text(json.dumps(v,indent=2)+"\n")
PY
sha256sum -c --quiet "$e/parent_executable.sha256"
tar -C "$work" -czf "$work/evidence.tgz" evidence
