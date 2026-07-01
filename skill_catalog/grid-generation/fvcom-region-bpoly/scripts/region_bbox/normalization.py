from __future__ import annotations

import re
from typing import Any


def request_text(request: dict[str, Any] | str) -> str:
    if isinstance(request, str):
        return request
    chunks = []
    for key in ("region", "mission", "purpose", "description", "request", "text", "prompt"):
        val = request.get(key)
        if val:
            chunks.append(str(val))
    if not chunks:
        chunks.append(" ".join(str(v) for v in request.values() if isinstance(v, (str, int, float))))
    return " ".join(chunks)


def normalize_request_text(request: dict[str, Any] | str) -> str:
    text = request_text(request).lower()
    text = text.replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-").replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\bsouth[\s-]*east\s+alaska\b", "southeast alaska", text)
    text = re.sub(r"\bsouth[\s-]*eastern\s+alaska\b", "southeast alaska", text)
    text = re.sub(r"\bse[\s-]*ak\b", "southeast alaska", text)
    text = re.sub(r"\balexander\s+archipelago\b", "southeast alaska alexander archipelago", text)
    text = re.sub(r"\blong\s+island\s+sound\b", "long island sound", text)
    text = re.sub(r"\bhawaii\s+island\b", "hawaii island", text)
    text = re.sub(r"\bhawaiian\s+islands\b", "hawaiian islands", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def canonical_region_key(request: dict[str, Any] | str) -> str:
    text = normalize_request_text(request)
    if "puget" in text or "salish" in text:
        return "puget_salish"
    if "long island sound" in text or "hypoxia" in text:
        return "long_island_sound"
    if "murderkill" in text:
        return "murderkill"
    if "delaware" in text:
        return "delaware"
    if "chesapeake" in text:
        return "chesapeake"
    if "aleut" in text or "aleuc" in text:
        return "aleutian"
    if "hawaiian islands" in text or "hawaii islands" in text or "hawaii state" in text:
        return "hawaii_state"
    if "hawaii island" in text or "big island" in text or "otec" in text:
        return "hawaii_island"
    if "lake superior" in text:
        return "lake_superior"
    if "lake ontario" in text:
        return "lake_ontario"
    if "lake erie" in text:
        return "lake_erie"
    if "cook inlet" in text:
        return "cook_inlet"
    if "southeast alaska" in text:
        return "southeast_alaska"
    if "columbia" in text:
        return "columbia"
    if "hudson" in text:
        return "hudson"
    if "san francisco" in text:
        return "san_francisco"
    return "unknown"

