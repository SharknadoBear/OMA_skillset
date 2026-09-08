"""Decode browser_gif_background.js artifacts and reject stationary-pixel flicker.

Requires Pillow and NumPy for QA only; the viewer/exporter has no Pillow dependency.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

def verify(gif, mask, report):
    meta=json.loads(Path(report).read_text(encoding="utf-8"))
    selection=np.fromfile(mask,dtype=np.uint8).reshape(meta["height"],meta["width"])
    stationary=selection>0;water=selection==3
    if water.sum()<10000:raise ValueError("Insufficient unchanged shaded-water pixels; use a wider wet view")
    with Image.open(gif) as image:
        if image.size!=(meta["width"],meta["height"]) or image.n_frames!=meta["frames"]:raise ValueError("GIF dimensions/frame count differ from report")
        if image.info.get("loop")!=0:raise ValueError("GIF must repeat infinitely")
        reference=np.array(image.convert("RGB"))[stationary];max_difference=0
        for frame in range(image.n_frames):
            image.seek(frame)
            if image.info.get("duration")!=1000/meta["fps"]:raise ValueError("Incorrect frame delay")
            actual=np.array(image.convert("RGB"))[stationary]
            difference=int(np.abs(actual.astype(np.int16)-reference.astype(np.int16)).max())
            max_difference=max(max_difference,difference)
            if difference:raise ValueError(f"Frame {frame}: unchanged source pixels changed by {difference} color levels")
    return dict(passed=True,frames=meta["frames"],stationary_source_pixels=int(stationary.sum()),stationary_water_pixels=int(water.sum()),max_static_pixel_difference=max_difference)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["gif","mask","report"]:parser.add_argument("--"+name,required=True)
    args=parser.parse_args();print(json.dumps(verify(args.gif,args.mask,args.report),indent=2))
