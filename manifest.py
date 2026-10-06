"""Builds slides.csv from a live listing of the public CAMELYON16 bucket.

Nothing about the bucket is assumed beyond its address: file names, sizes and checksums are
read from the listing, test labels from evaluation/reference.csv. The lesion annotations
(small XML files) are downloaded into data/annotations/.

    python manifest.py
"""
import csv
import io
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import budget
import config

COLUMNS = ["slide", "split", "label", "lesion", "key", "bytes", "md5", "annotation"]


def fetch(key: str, query: str = "") -> bytes:
    url = f"{config.BUCKET_URL}/{urllib.parse.quote(key)}{query}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def parse_listing(xml_bytes: bytes) -> tuple[list[tuple[str, int]], str | None]:
    """One page of an S3 ListObjectsV2 reply -> ([(key, size)], continuation token or None)."""
    root = ET.fromstring(xml_bytes)
    ns = root.tag[: root.tag.index("}") + 1] if root.tag.startswith("{") else ""
    items = [(c.findtext(f"{ns}Key"), int(c.findtext(f"{ns}Size"))) for c in root.iter(f"{ns}Contents")]
    return items, root.findtext(f"{ns}NextContinuationToken")


def list_bucket(prefix: str) -> dict[str, int]:
    out, token = {}, None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if token:
            q["continuation-token"] = token
        items, token = parse_listing(fetch("", "?" + urllib.parse.urlencode(q)))
        out.update(items)
        if not token:
            return out


def parse_checksums(text: str) -> dict[str, str]:
    """`<md5>  <path>` lines -> {file name: md5}."""
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and re.fullmatch(r"[0-9a-fA-F]{32}", parts[0]):
            out[Path(parts[-1].lstrip("*")).name] = parts[0].lower()
    return out


def parse_reference(text: str) -> dict[str, tuple[str, str]]:
    """evaluation/reference.csv -> {slide: (label, lesion)}; label is "normal" or "tumor".

    Works with or without a header row. The lesion column (macro, micro, itc, negative) is
    kept when present.
    """
    out = {}
    for row in csv.reader(io.StringIO(text)):
        cells = [c.strip() for c in row]
        if not cells or not re.fullmatch(r"test_\d+(\.tif)?", cells[0], flags=re.I):
            continue
        rest = [c.lower() for c in cells[1:]]
        if "tumor" in rest:
            label = "tumor"
        elif "normal" in rest or "negative" in rest:
            label = "normal"
        else:
            raise ValueError(f"cannot read a label from reference row: {row}")
        lesion = next((c for c in rest if c in ("macro", "micro", "itc", "negative", "none")), "")
        out[cells[0].lower().removesuffix(".tif")] = (label, lesion)
    return out


def build_rows(keys: dict[str, int], md5: dict[str, str], reference: dict[str, tuple[str, str]],
               annotations: set[str]) -> list[dict]:
    rows = []
    for key, size in sorted(keys.items()):
        m = re.fullmatch(r".*/images/((normal|tumor|test)_\d+)\.tif", key, flags=re.I)
        if not m:
            continue
        slide, kind = m.group(1).lower(), m.group(2).lower()
        if kind == "test":
            if slide not in reference:
                print(f"warning: {slide} has no entry in reference.csv, skipped")
                continue
            split, (label, lesion) = "test", reference[slide]
        else:
            split, label, lesion = "train", kind, ""
        rows.append(dict(slide=slide, split=split, label=label, lesion=lesion, key=key, bytes=size,
                         md5=md5.get(Path(key).name, ""), annotation=slide if slide in annotations else ""))
    return rows


def download_annotations(keys: dict[str, int]) -> set[str]:
    """Fetch every lesion annotation into data/annotations/<slide>.xml. Returns the slide names."""
    config.ANNOTATIONS.mkdir(parents=True, exist_ok=True)
    wanted = {k: s for k, s in keys.items() if "/annotations/" in k and k.lower().endswith((".xml", ".zip"))}
    budget.reserve(2 * sum(wanted.values()), "lesion annotations")
    for key in sorted(wanted):
        name = Path(key).name.lower()
        if name.endswith(".xml"):
            target = config.ANNOTATIONS / name
            if not target.exists():
                target.write_bytes(fetch(key))
        else:
            with zipfile.ZipFile(io.BytesIO(fetch(key))) as z:
                for member in z.namelist():
                    if member.lower().endswith(".xml"):
                        (config.ANNOTATIONS / Path(member).name.lower()).write_bytes(z.read(member))
    return {p.stem for p in config.ANNOTATIONS.glob("*.xml")}


def main() -> None:
    keys = list_bucket(config.PREFIX)
    print(f"{len(keys)} objects under {config.PREFIX}")
    md5 = parse_checksums(fetch(config.PREFIX + "checksums.md5").decode())
    ref_key = next((k for k in keys if k.lower().endswith("evaluation/reference.csv")), None)
    if ref_key is None:
        raise SystemExit("evaluation/reference.csv not found in the bucket listing; test labels unavailable")
    reference = parse_reference(fetch(ref_key).decode())
    annotations = download_annotations(keys)

    rows = build_rows(keys, md5, reference, annotations)
    with open(config.MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    def count(split, label):
        return sum(r["split"] == split and r["label"] == label for r in rows)

    total = sum(r["bytes"] for r in rows)
    print(f"{len(rows)} slides, {total / budget.GB:.0f} GB to download in total (never more than one on disk)")
    print(f"  train: {count('train', 'normal')} normal, {count('train', 'tumor')} tumour")
    print(f"  test:  {count('test', 'normal')} normal, {count('test', 'tumor')} tumour")
    missing = [r["slide"] for r in rows if r["label"] == "tumor" and not r["annotation"]]
    if missing:
        print(f"warning: {len(missing)} tumour slides have no annotation file: {missing[:5]}")
    if not all(r["md5"] for r in rows):
        print("warning: some slides have no checksum; their downloads will be checked by size only")
    print(f"wrote {config.MANIFEST.name}")


if __name__ == "__main__":
    main()
