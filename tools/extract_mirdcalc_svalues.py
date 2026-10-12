#!/usr/bin/env python3
"""Extract a small organ S-value dataset from the MIRDcalc S-value database.

Why this exists: the shipped scenario needs cross-organ S-values, but the MIRDcalc database is a
548 MB archive with its own terms of use (personal / non-commercial; see docs/data-sources.md).
This script reads only the one CSV it needs (HTTP range requests on a URL, or a local zip),
verifies its CRC-32, and writes a few dozen numbers plus full provenance, so the dataset in this
repository can be regenerated and audited by anyone who obtains the source themselves.

    python -I tools/extract_mirdcalc_svalues.py \
        --source https://mirdsoft.org/products/MIRDcalc/MIRDcalcDB-v20251023-1224x12-csv.zip \
        --out src/RadioNuclide_Uncertainty_Lab/data/lu177_adult_male.json

Standard library only. Run it with ``-I`` and never point it at a file you did not obtain
yourself; it parses text and executes nothing from the source.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

MGY_PER_MBQ_S_TO_GY_PER_MBQ_H = 3600.0 / 1000.0  # exact
DEFAULT_REGIONS = ["Kidneys", "Liver", "Spleen", "Bone marrow - red (active)"]
CITATION = (
    "Kesner AL, Carter LM, Ocampo Ramos JC, et al. MIRD Pamphlet No. 28, Part 1: MIRDcalc - a "
    "freely available software tool for patient-specific internal radiation dose assessment "
    "(J Nucl Med 2023;64(10):1524-1531, doi:10.2967/jnumed.122.264225), as requested on the "
    "database page https://mirdsoft.org/s-value-database"
)
TERMS = (
    "Source terms (mirdsoft.org): content is copyright protected and may be used for personal "
    "and noncommercial use without prior written consent; no open licence is stated for the "
    "dataset. Only the numbers listed here are retained. Confirm the terms before "
    "redistributing; regenerate with this tool from your own copy if in doubt."
)


class RangeFile(io.RawIOBase):
    """Read-only seekable file over HTTP range requests (so a zip need not be downloaded)."""

    def __init__(self, url: str):
        self.url, self.pos, self.bytes_read = url, 0, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as r:
            self.size = int(r.headers["Content-Length"])
            self.last_modified = r.headers.get("Last-Modified")

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n < 0 or self.pos + n > self.size:
            n = self.size - self.pos
        if n <= 0:
            return b""
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"}
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        self.pos += len(data)
        self.bytes_read += len(data)
        return data

    def readinto(self, b) -> int:
        d = self.read(len(b))
        b[: len(d)] = d
        return len(d)


def parse_sheet(text: str, regions: list[str]) -> dict:
    """Parse one MIRDcalc S-value CSV. Rows = source regions, columns = target regions."""
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or rows[0][0] != "Phantom":
        raise ValueError("unexpected file: first cell is not 'Phantom'")
    meta = {"phantom": rows[0][1], "db_version": rows[0][rows[0].index("DB version") + 1]}

    # mass table: source tissue mass (col 3) / blood (col 4); target tissue-and-blood mass (col 13)
    hdr = next(r for r in rows if len(r) > 13 and r[2] == "SOURCE ORGAN")
    assert (
        hdr[3].startswith("TISSUE MASS")
        and hdr[12] == "TARGET ORGAN"
        and hdr[13].startswith("TISSUE & BLOOD")
    )
    masses_source, masses_target = {}, {}
    for r in rows[rows.index(hdr) + 1 :]:
        if len(r) > 13 and r[2] in regions:
            masses_source[r[2]] = {"tissue_g": float(r[3]), "blood_g": float(r[4])}
        if len(r) > 13 and r[12] in regions:
            masses_target[r[12]] = float(r[13])
        if r and r[0] == "Isotope":
            break

    blocks: dict[str, dict] = {}
    i = 0
    while i < len(rows):
        if len(rows[i]) > 1 and rows[i][0] == "Radiation":
            radiation, i = rows[i][1], i + 1
            while not rows[i][0].startswith("Organ S-value"):
                i += 1
            unit_line = rows[i][0]
            if "mGy/(MBq s)" not in unit_line:
                raise ValueError(f"unexpected S-value units: {unit_line!r}")
            targets = [c for c in rows[i + 1][1:] if c.strip()]
            j, matrix = i + 2, {}
            while j < len(rows) and rows[j] and rows[j][0].strip():
                if rows[j][0] in regions:
                    matrix[rows[j][0]] = dict(
                        zip(targets, (float(v) for v in rows[j][1 : 1 + len(targets)]), strict=True)
                    )
                j += 1
            blocks[radiation] = matrix
            i = j
        else:
            i += 1
    for needed in ("Gamma", "Beta", "Alpha"):
        if needed not in blocks:
            raise ValueError(f"missing {needed} block")
    return {
        "meta": meta,
        "blocks": blocks,
        "masses_source": masses_source,
        "masses_target": masses_target,
    }


def build(sheet: dict, regions: list[str], dataset: str, nuclide: str, provenance: dict) -> dict:
    conv = MGY_PER_MBQ_S_TO_GY_PER_MBQ_H
    missing = [
        r for r in regions if r not in sheet["masses_target"] or r not in sheet["blocks"]["Gamma"]
    ]
    if missing:
        raise ValueError(f"regions not found: {missing}")
    total, photon = {}, {}
    for t in regions:
        total[t], photon[t] = {}, {}
        for s in regions:
            parts = {k: sheet["blocks"][k][s][t] for k in ("Gamma", "Beta", "Alpha")}
            total[t][s] = sum(parts.values()) * conv
            photon[t][s] = parts["Gamma"] * conv
    return {
        "dataset": dataset,
        "nuclide": nuclide,
        "phantom": sheet["meta"]["phantom"],
        "units": {"s_value": "Gy/(MBq h)", "mass": "kg"},
        "orientation": "s_total[target][source] = S(target <- source)",
        "regions": regions,
        "mass_kg": {r: sheet["masses_target"][r] / 1000.0 for r in regions},
        "mass_note": "target 'TISSUE & BLOOD MASS' of the phantom (the mass used for self-dose S-values)",
        "s_total": total,
        "s_photon": photon,
        "provenance": {
            **provenance,
            "db_version_in_file": sheet["meta"]["db_version"],
            "citation": CITATION,
            "terms": TERMS,
            "decay_data": "ICRP Publication 107 (as stated by the database)",
            "phantom_reference": "ICRP Publication 110 (as stated by the database)",
            "extraction_tool": "tools/extract_mirdcalc_svalues.py",
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--source", required=True, help="URL of, or local path to, the MIRDcalc CSV zip")
    p.add_argument("--nuclide", default="Lu-177")
    p.add_argument("--phantom", default="ICRP Adult Male")
    p.add_argument(
        "--regions", default=",".join(DEFAULT_REGIONS), help="comma-separated region names"
    )
    p.add_argument("--dataset", default="lu177_adult_male")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args(argv)
    regions = [r.strip() for r in a.regions.split(",")]

    if a.source.startswith(("http://", "https://")):
        fobj = RangeFile(a.source)
        archive = {
            "archive_url": a.source,
            "archive_bytes": fobj.size,
            "archive_last_modified": fobj.last_modified,
        }
    else:
        fobj = open(a.source, "rb")  # noqa: SIM115
        path = Path(a.source)
        archive = {"archive_path_name": path.name, "archive_bytes": path.stat().st_size}
    zf = zipfile.ZipFile(fobj)
    member = next(
        (n for n in zf.namelist() if n.endswith(f"SDATA_{a.nuclide}_{a.phantom}.csv")), None
    )
    if member is None:
        print(f"no entry for {a.nuclide} / {a.phantom}", file=sys.stderr)
        return 2
    info = zf.getinfo(member)
    data = zf.read(info)  # zipfile verifies the CRC-32
    sheet = parse_sheet(data.decode("utf-8-sig"), regions)
    provenance = {
        **archive,
        "archive_member": member,
        "member_crc32": f"{info.CRC:08x}",
        "member_sha256": hashlib.sha256(data).hexdigest(),
        "member_bytes": len(data),
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bytes_fetched": getattr(fobj, "bytes_read", None),
    }
    out = build(sheet, regions, a.dataset, a.nuclide, provenance)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(
        f"wrote {a.out} ({len(regions)}x{len(regions)} regions; fetched "
        f"{provenance['bytes_fetched']} bytes of {archive['archive_bytes']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
