"""Apply a read-only salt-budget instrument to the qualified Cartesian FVCOM_MP source."""
import argparse
import hashlib
import json
from pathlib import Path

SOURCE_SHA256 = {
    "internal_step.F": "7f6e41e51186c3cb8586d138957fefa07bee146a481f8a088ca72d935b98d1b9",
    "adv_s.F": "dc782c6eeaf17a038164e30d9eb613c0f933731457957b00a73d345ae63da3ee",
    "bcond_ts.F": "3f4bab1d2dcceecc08c3783eacbdd06bb87fccfba329d73a890e478a139ed8be",
    "mod_station_timeseries.F": "07f9a4fef34d1d89e07487ea7399f8221e2a9c521720c011108397f1ee155000",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def instrument(source, ledger, receipt):
    before = {name: sha(source / name) for name in SOURCE_SHA256}
    if before != SOURCE_SHA256:
        raise ValueError("Expected the qualified f8e4b50 Perlmutter GNU Cartesian source")
    target = source / "internal_step.F"
    lines = target.read_text().splitlines(keepends=True)
    anchors = [
        ("CALL ADV_S", ["call estuary_salt_stage(1,S1,DT,ISWETNT)"], ["call estuary_salt_stage(2,SF1,DTFA,ISWETN)"]),
        ("IF(SCALAR_POSITIVITY_CONTROL) CALL FCT_S", ["call estuary_salt_stage(3,SF1,DTFA,ISWETN)"], ["call estuary_salt_stage(4,SF1,DTFA,ISWETN)"]),
        ("IF(PAR)CALL AEXCHANGE(NC,MYID,NPROCS,SF1)", ["call estuary_salt_stage(5,SF1,DTFA,ISWETN)"], ["call estuary_salt_stage(6,SF1,DTFA,ISWETN)"]),
        ("CALL BCOND_TS(2)", ["call estuary_salt_boundary(1,SF1)"], ["call estuary_salt_boundary(2,SF1)", "call estuary_salt_stage(7,SF1,DTFA,ISWETN)"]),
        ("S1 = SF1", [], ["call estuary_salt_stage(8,S1,D,ISWETN)"]),
    ]
    for anchor, pre, post in anchors:
        indices = [i for i,line in enumerate(lines) if line.split("!",1)[0].strip() == anchor]
        if len(indices) != 1:
            raise ValueError("Expected one native anchor: " + anchor)
        i = indices[0]
        lines[i:i+1] = ["     "+line+"\n" for line in pre] + [lines[i]] + ["     "+line+"\n" for line in post]
    with target.open("w",newline="\n") as stream:
        stream.write("".join(lines)+"\n"+ledger.read_text())
    station = source / "mod_station_timeseries.F"
    station_text = station.read_text()
    for old, new in [
        ("integer,private :: el_s_vid", "integer,private :: wet_s_vid\n   integer,private :: el_s_vid"),
        ("SUBROUTINE OUT_STATION_TIMESERIES \n", "SUBROUTINE OUT_STATION_TIMESERIES \n   use mod_wd, only: iswetc\n"),
        ("!--Write Variables to File", """!--Write Variables to File
     if (trim(location_type)=='cell' .and. out_salt_temp) then
       i1=lbound(iswetc,1)
       i2=ubound(iswetc,1)
       call putvar(i1,i2,n,ngl,1,1,'e',iswetc,nc_ofid,wet_s_vid,myid,nprocs,ipt,stck_cnt)
     endif"""),
        ('ierr = nf90_def_var(nc_ofid,"zeta",nf90_float,dynm2ds,el_s_vid)', '''if (trim(location_type)=='cell' .and. out_salt_temp) then
       ierr=nf90_def_var(nc_ofid,"wet_cells",nf90_int,dynm2ds,wet_s_vid)
       call handle_ncerr(ierr,"define station cell wet mask")
       ierr=nf90_put_att(nc_ofid,wet_s_vid,"long_name","native wet cell flag at station")
     endif
     ierr = nf90_def_var(nc_ofid,"zeta",nf90_float,dynm2ds,el_s_vid)'''),
    ]:
        if station_text.count(old)!=1:
            raise ValueError("Unexpected native station output anchor: "+old)
        station_text=station_text.replace(old,new)
    station.write_text(station_text,newline="\n")
    data = {"schema":"fvcom_salinity_stage_instrument_v1","source_commit":"f8e4b50d4fd51409febcefc763cbc5dca11870e2",
            "original_sources":before,"instrumented_internal_step_sha256":sha(target),"ledger_sha256":sha(ledger),
            "instrumented_station_writer_sha256":sha(station),
            "salinity_transport":"unmodified ADV_S, FCT_S, VDIF_TS and BCOND_TS",
            "storage_units":"practical salinity times cubic metres",
            "rank_ownership":"primary nodes 1:M and inverse BN_MLT on shared primary nodes",
            "stage_columns":["ticks","iint","first_volume","volume","first_salt","salt","between_steps","advection_and_volume","node_match","FCT_S","vertical_diffusion","halo_exchange","BCOND_TS","final_geometry"],
            "boundary_columns":["ticks","iint","outward_m3","inward_m3","outward_salt","inward_source_salt","outward_update","inward_update","outward_weighted_steps","inward_weighted_steps","inward_applied_salt","river_m3","river_salt","max_inward_source_error"]}
    receipt.write_text(json.dumps(data,indent=2)+"\n")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("ledger",type=Path)
    parser.add_argument("receipt",type=Path)
    args=parser.parse_args()
    instrument(args.source,args.ledger,args.receipt)
