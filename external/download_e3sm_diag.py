#!/usr/bin/env python3
from pathlib import Path
import requests

OWNER = "E3SM-Project"
REPO = "e3sm_diags"
BRANCH = "main"
DIR = "e3sm_diags/driver/control_runs"

OUTDIR = Path("./")
OUTDIR.mkdir(exist_ok=True)

api_url = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{DIR}?ref={BRANCH}"

resp = requests.get(api_url, timeout=60)
resp.raise_for_status()

files = resp.json()

for item in files:
    if item["type"] != "file":
        continue

    filename = item["name"]
    download_url = item["download_url"]

    outpath = OUTDIR / filename
    print(f"Downloading {filename}")

    r = requests.get(download_url, timeout=60)
    r.raise_for_status()
    outpath.write_bytes(r.content)

print(f"Done. Downloaded files to: {OUTDIR.resolve()}")

