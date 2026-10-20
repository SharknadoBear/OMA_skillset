# Read-only login-node reconnaissance; no scheduler submissions or env-file edits.
export TERM=dumb
export LMOD_PAGER=none
export PAGER=cat
set -e -o pipefail
printf '\n== Identity and paths ==\n'
whoami
hostname
pwd
printf 'NERSC_HOST=%s\nHOME=%s\nSCRATCH=%s\nPSCRATCH=%s\nCFS=%s\n' "$NERSC_HOST" "$HOME" "$SCRATCH" "$PSCRATCH" "$CFS"
for p in "$HOME" "$SCRATCH"; do
    if [ -n "$p" ]; then
        ls -ld -- "$p"
        readlink -f -- "$p"
        df -h -- "$p"
    fi
done
printf '\n== Login architecture ==\n'
uname -sm
printf '\n== Module command ==\n'
type module
module --version 2>&1
printf '\n== Default loaded modules ==\n'
module -t list 2>&1
printf '\n== Compiler and build executables ==\n'
for command_name in cc CC ftn gcc gfortran nvcc cmake python3; do
    command -v "$command_name" || true
done
printf '\n== Relevant module availability ==\n'
for module_name in PrgEnv-gnu PrgEnv-nvidia gcc nvidia cudatoolkit cmake cray-hdf5-parallel cray-netcdf-hdf5parallel; do
    printf '\n-- %s --\n' "$module_name"
    if module -t avail "$module_name" 2>&1; then
        printf 'module_query_exit=0\n'
    else
        printf 'module_query_exit=%s\n' "$?"
    fi
done
printf '\n== Quota tools ==\n'
command -v showquota || true
printf '\n== Probe complete ==\n'
