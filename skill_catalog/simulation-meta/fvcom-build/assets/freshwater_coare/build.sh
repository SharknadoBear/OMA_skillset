#!/bin/bash -l
#SBATCH --job-name=fvcom_lake_coare_build
#SBATCH --account=@ACCOUNT@
#SBATCH --qos=@QOS@
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --output=build_%j.log
set -eo pipefail
module reset
if module is-loaded gpu; then module swap gpu cpu; else module load cpu; fi
if module is-loaded cudatoolkit; then module unload cudatoolkit; fi
module load @MODULES@
set -u
payload=@WORK@
base=@BASE@
previous=@EVIDENCE@
cd "$payload"
test "$NERSC_HOST" = perlmutter
sha256sum -c --quiet inputs.sha256
py=/usr/bin/python3.11
"$py" freshwater_coare.py check-baseline --source "$base" --evidence "$previous" --binding base_binding.json
(cd "$base"; sha256sum -c --quiet "$previous/patched_sources.sha256")
out="$payload/build_$SLURM_JOB_ID"; src="$out/source"; e="$out/evidence"
mkdir "$out" "$src" "$e"
tar -C "$base" --exclude='./.git' --exclude='./fvcom_plased' --exclude='./*.o' --exclude='./*.mod' -cf - . | tar -C "$src" -xf -
module -t list > "$e/modules.txt" 2>&1
scontrol show job -o "$SLURM_JOB_ID" > "$e/allocation.txt"
"$py" "$payload/patch_coare.py" "$src" "$e/changes.json"
@DONOR_TRANSFORM@
cd "$src"
"$py" - "$base" "$previous" "$e" <<'PY'
import json,sys
from pathlib import Path
base,old,e=map(Path,sys.argv[1:])
make=Path("make.inc");text=make.read_text();key="TOPDIR = "+str(base)
if text.count(key)!=1:raise ValueError("baseline TOPDIR mismatch")
make.write_text(text.replace(key,"TOPDIR = "+str(Path.cwd()))+"\nMODEL_FLAGS += -DHEATING_CALCULATED -DLAKE_COARE_DIAG\n")
a=json.loads((old/"make_inc_audit.json").read_bytes())
a["active_defines"]+=["HEATING_CALCULATED","LAKE_COARE_DIAG"]
(e/"active_defines.json").write_text(json.dumps(a,indent=2)+"\n")
PY
cp make.inc "$e/make.inc"
cp "$previous/compiled_objects.txt" "$e/objects.txt"
cp "$previous/source_revision.json" "$e/"
"$py" "$payload/perlmutter_dependencies.py" "$e/active_defines.json" "$e/objects.txt" > "$e/dependencies.mk"
make -k -f makefile -f "$e/dependencies.mk" -j16 fvcom_plased > "$e/build.log" 2>&1
sha256sum fvcom_plased > "$e/executable.sha256"
ldd fvcom_plased > "$e/ldd.txt"
mkdir "$out/template"
(cd "$out/template"; srun --exact --ntasks=1 --cpus-per-task=1 --cpu-bind=cores "$src/fvcom_plased" --create_namelist=blank > "$e/blank_probe.log" 2>&1)
cp "$out/template/blank_run.nml" "$e/blank_namelist.nml"
"$py" "$payload/freshwater_coare.py" freeze --source "$src" --evidence "$e" --binding "$payload/base_binding.json" --job-id "$SLURM_JOB_ID" @DONOR_OPTION@ > "$e/coare_build_freeze.json"
tar -C "$out" -czf "$payload/evidence_$SLURM_JOB_ID.tgz" evidence
