"""Finalize a reviewed attempt without rewriting its hash-bound evidence."""
import html
import re
import shutil
from pathlib import Path


def finalize(case_root, delivery, release_revision):
    from remora_grid import validate_grid, readback, read_json, write_json, resolve, sha, file_record
    root=Path(case_root).resolve()
    delivery=Path(delivery).resolve()
    relative=delivery.relative_to(root)
    if (len(relative.parts)!=4 or relative.parts[0]!="attempts" or
        not re.fullmatch(r"attempt_\d{3,}",relative.parts[1]) or
        relative.parts[2:] != ("03_grid","grid_delivery.json")):
        raise ValueError("Delivery must be case/attempts/attempt_NNN/03_grid/grid_delivery.json")
    if release_revision != "unpublished" and not re.fullmatch(r"[0-9a-f]{40}",release_revision):
        raise ValueError("Provide the published commit SHA, or unpublished for development fixtures")
    request=root/"science_request.json"
    science=read_json(request)
    doc=validate_grid(delivery,reviewed=True)
    for name,digest in doc["software"]["scripts_sha256"].items():
        if sha(Path(__file__).parent/name)!=digest:
            raise ValueError("Installed code changed since build: "+name)
    source=resolve(delivery.parent,doc["grid"]["path"])
    validation=read_json(resolve(delivery.parent,doc["validation"]["path"]))
    checked=readback(source)
    if validation["smoothing"].get("method")!="area_weighted_log_pair_projection_v1":
        raise ValueError("Final delivery requires log-depth smoothing evidence")
    fit_path=resolve(delivery.parent,doc["fit"]["path"])
    fit=read_json(fit_path)
    region_path=resolve(fit_path.parent,fit["region_delivery"]["path"])
    region=read_json(region_path)
    named={Path(m["path"]).stem:resolve(delivery.parent,m["path"]) for m in doc["maps"]}
    sections=[("Scientific polygon and required features",[("region",resolve(region_path.parent,m["path"])) for m in region["maps"]]),
              ("Fitted footprint and grid geometry",[("fit",resolve(fit_path.parent,m["path"])) for m in fit["maps"]]+[("grid",named["geometry_quality"])]),
              ("Initial and final masks and corrected cells",[("grid",named["mask_changes"])]),
              ("Bathymetry before and after smoothing",[("grid",named["bathymetry_comparison"])]),
              ("Absolute and relative depth changes",[("grid",named["smoothing_changes"])]),
              ("Vertical sections and required-feature closeups",[("grid",named["vertical_sections"])]+[("grid",v) for k,v in named.items() if k.startswith("feature_")])]
    target=root/"final"
    if target.exists():
        raise ValueError("Final delivery already exists; preserve it and use a new case revision")
    target.mkdir()
    (target/"maps").mkdir()
    shutil.copyfile(source,target/"remora_grid.nc")
    if sha(target/"remora_grid.nc")!=doc["grid"]["sha256"]:
        raise ValueError("Final NetCDF copy hash mismatch")
    shutil.copyfile(resolve(delivery.parent,doc["validation"]["path"]),target/"validation.json")
    panels=[]
    body=[]
    for heading,images in sections:
        block={"title":heading,"maps":[]}
        body.append("<h2>"+html.escape(heading)+"</h2>")
        for prefix,path in images:
            dest=target/"maps"/(prefix+"_"+path.name)
            shutil.copyfile(path,dest)
            if sha(dest)!=sha(path): raise ValueError("Map copy hash mismatch")
            rec=file_record(dest,target)
            block["maps"].append(rec)
            body.append(f'<figure><img src="{html.escape(rec["path"].replace(chr(92),"/"))}" alt="{html.escape(path.stem)}"><figcaption>{html.escape(path.stem.replace("_"," "))}</figcaption></figure>')
        panels.append(block)
    agent_review=read_json(delivery.parent/"grid_review.json")
    reader="not_run"
    if (delivery.parent/doc["reader_check"]).exists():
        validate_grid(delivery,reviewed=True,reader=True)
        reader="pass"
    report={"schema":"remora_case_delivery_v1","case_id":root.name,"selected_attempt":relative.parts[1],
            "grid":file_record(target/"remora_grid.nc",target),"original_delivery":file_record(delivery,target),
            "science_request":file_record(request,target),"release_revision":release_revision,
            "software":doc["software"],"local_validation":checked["local_validation"],
            "validation":file_record(target/"validation.json",target),
            "agent_review":file_record(delivery.parent/"grid_review.json",target),
            "reader_validation":reader,"human_review":"pending","panels":panels}
    title=science.get("location",science.get("case_id",root.name))
    esc=lambda x:html.escape(str(x))
    purpose=science.get("modeling_purpose",science.get("objective",""))
    warnings="".join("<li>"+esc(x)+"</li>" for x in validation.get("warnings",[]))
    smoothing=validation["smoothing"]
    page=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)} REMORA grid review</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;max-width:1400px;margin:2rem auto;padding:0 1.2rem;color:#183247}}h2{{margin-top:2.5rem}}img{{width:100%;height:auto}}figure{{margin:1rem 0}}a{{color:#006897}}pre{{white-space:pre-wrap}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccd8df;padding:.5rem;text-align:left}}</style></head><body>
<h1>{esc(title)} — REMORA grid review</h1><p>{esc(purpose)}</p>
<p><a href="remora_grid.nc">Final NetCDF</a> · <a href="case_delivery.json">Case manifest</a> · <a href="validation.json">Numerical diagnostics</a></p>
<p>Human review: pending. Agent review: {esc(agent_review['decision'])}. Local validation: pass. Executable reader: {esc(reader)}. Simulation validation: not run.</p>
<p>Grid: {esc(checked['n_cell'])}; spacing {fit['parameters']['spacing_m']:g} m. Smoothing: log depth; rx0 {checked['rx0']:.8f}. Physical-interior volume change: {smoothing['physical_volume_relative_change']*100:.3f}%.</p>
<p>Agent assessment: {esc(agent_review['rationale'])}</p><ul>{warnings}</ul>
{''.join(body)}<h2>Feature-section diagnostics</h2><p>Grid-axis transects through protected wet features; these are not automatically channel-normal sections.</p><pre>{esc(__import__('json').dumps(validation.get('feature_sections',[]),indent=2))}</pre></body></html>'''
    (target/"review.html").write_text(page,encoding="utf-8")
    report["review_page"]=file_record(target/"review.html",target)
    write_json(target/"case_delivery.json",report)
    validate_case(target/"case_delivery.json")
    return target/"case_delivery.json"


def validate_case(path):
    from remora_grid import read_json, resolve, sha, readback, validate_grid
    path=Path(path).resolve(); d=read_json(path)
    if d.get("schema")!="remora_case_delivery_v1": raise ValueError("Unsupported case delivery")
    records=[d[k] for k in ["grid","original_delivery","science_request","validation","agent_review","review_page"]]
    records += [m for p in d["panels"] for m in p["maps"]]
    for rec in records:
        if sha(resolve(path.parent,rec["path"]))!=rec["sha256"]:
            raise ValueError("Changed case evidence: "+rec["path"])
    original=validate_grid(resolve(path.parent,d["original_delivery"]["path"]),reviewed=True,
                           reader=d["reader_validation"]=="pass")
    if d["grid"]["sha256"]!=original["grid"]["sha256"]: raise ValueError("Final grid differs from selected attempt")
    readback(resolve(path.parent,d["grid"]["path"]))
    return d
