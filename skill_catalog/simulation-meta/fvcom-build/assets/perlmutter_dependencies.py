"""Order the active FVCOM Fortran module producers before their USE sites."""
import argparse
import json
from pathlib import Path
import re
import subprocess


def dependencies(sources):
    providers = {}
    uses = {}
    for name, body in sources:
        target = Path(name).with_suffix(".o").name
        uses[target] = set()
        for line in body.splitlines():
            code = line.partition("!")[0].strip().lower()
            definition = re.fullmatch(r"module\s+(\w+)", code)
            if definition:
                module = definition[1]
                if module in providers and providers[module] != target:
                    raise ValueError(f"Duplicate module {module} in {name}")
                providers[module] = target
            use = re.match(r"use\s*(?:,\s*non_intrinsic\s*)?(?:::\s*)?\b([a-z]\w*)\b", code)
            if use:
                uses[target].add(use[1])
    return {target: sorted({providers[module] for module in modules if module in providers} - {target})
            for target, modules in uses.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("objects", type=Path)
    args = parser.parse_args()
    flags = ["-D"+name for name in json.loads(args.audit.read_text())["active_defines"]]
    sources = []
    for obj in args.objects.read_text().split():
        for suffix in (".F", ".F90"):
            source = Path(obj).with_suffix(suffix)
            if source.is_file():
                process = subprocess.run(["/usr/bin/cpp", "-P", "-traditional", "-nostdinc", *flags, str(source)],
                                         stdout=subprocess.PIPE, check=True)
                sources.append((source.name, process.stdout.decode(errors="replace")))
                break
    for target, prereqs in sorted(dependencies(sources).items()):
        if prereqs:
            print(target+":", *prereqs)
