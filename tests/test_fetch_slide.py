import json

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
pytest.importorskip("tiffslide")

import config
import fetch_slide

W0, H0 = 4096, 2816
TISSUE = (slice(512, 2304), slice(896, 3584))   # rows, cols of the stained rectangle at level 0


def make_slide(path):
    """A small pyramidal TIFF: white glass with one textured pink rectangle, six levels."""
    rng = np.random.default_rng(0)
    base = np.full((H0, W0, 3), 245, dtype=np.uint8)
    block = rng.integers(0, 40, size=(TISSUE[0].stop - TISSUE[0].start, TISSUE[1].stop - TISSUE[1].start, 1))
    base[TISSUE] = np.array([200, 90, 160]) + block
    levels = [base] + [base[:: 2 ** i, :: 2 ** i] for i in range(1, 6)]
    with tifffile.TiffWriter(path) as tif:
        tif.write(levels[0], subifds=len(levels) - 1, tile=(256, 256), photometric="rgb", compression="deflate")
        for lvl in levels[1:]:
            tif.write(lvl, subfiletype=1, tile=(256, 256), photometric="rgb", compression="deflate")
    return base


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    for name, sub in (("SLIDES", "slides"), ("CACHE", "cache"), ("TILES", "tiles"),
                      ("EMBEDDINGS", "embeddings"), ("ANNOTATIONS", "annotations")):
        monkeypatch.setattr(config, name, tmp_path / "data" / sub)
        (tmp_path / "data" / sub).mkdir(parents=True)
    return tmp_path


ROW = dict(slide="tumor_901", split="train", label="tumor", lesion="", key="x", bytes="0", md5="")


def test_tile_slide_outputs(project):
    slide_file = config.SLIDES / "tumor_901.tif"
    base = make_slide(slide_file)
    # A lesion covering the tile whose corner is (896, 896), and half of the one to its right.
    (config.ANNOTATIONS / "tumor_901.xml").write_text(
        '<ASAP_Annotations><Annotations><Annotation PartOfGroup="Tumor"><Coordinates>'
        '<Coordinate Order="0" X="896" Y="896"/><Coordinate Order="1" X="2240" Y="896"/>'
        '<Coordinate Order="2" X="2240" Y="1792"/><Coordinate Order="3" X="896" Y="1792"/>'
        "</Coordinates></Annotation></Annotations></ASAP_Annotations>")

    n = fetch_slide.tile_slide(slide_file, ROW)

    z = np.load(fetch_slide.tiles_path("tumor_901"))
    coords, tumour = z["coords"], z["tumour"].astype(float)
    meta = json.loads(str(z["meta"]))
    # Tile is 896 px at level 0. The rectangle spans x 896..3584 (3 tiles) and y 512..2304:
    # rows at y=0 (43% tissue), y=896 (100%) and y=1792 (57%) all pass the 25% threshold.
    assert n == 9 == len(coords) == meta["n_tiles"]
    assert sorted(set(coords[:, 0].tolist())) == [896, 1792, 2688]
    assert sorted(set(coords[:, 1].tolist())) == [0, 896, 1792]
    assert meta["level_downsample"] == 4 and meta["annotated"] is True

    by_xy = {tuple(c): t for c, t in zip(coords.tolist(), tumour)}
    assert by_xy[(896, 896)] == pytest.approx(1.0, abs=0.05)
    assert by_xy[(1792, 896)] == pytest.approx(0.5, abs=0.06)
    assert by_xy[(2688, 896)] == 0 and by_xy[(896, 0)] == 0

    cache = np.load(fetch_slide.cache_path("tumor_901"), mmap_mode="r")
    assert cache.shape == (9, 224, 224, 3) and cache.dtype == np.uint8
    # Tile pixels are the level-0 pixels at every 4th position, in the same order as coords.
    i = [tuple(c) for c in coords.tolist()].index((1792, 896))
    assert np.array_equal(cache[i], base[896:1792:4, 1792:2688:4])
    assert (config.TILES / "tumor_901.jpg").exists()
    assert fetch_slide.is_cached("tumor_901")


def test_process_deletes_the_slide_and_is_idempotent(project):
    slide_file = config.SLIDES / "tumor_901.tif"
    make_slide(slide_file)
    assert fetch_slide.process(ROW) == 9
    assert not slide_file.exists()
    assert fetch_slide.process(ROW) == 9      # second call does nothing and needs no slide


def test_blank_slide_gives_no_tiles_and_no_cache(project):
    slide_file = config.SLIDES / "normal_901.tif"
    blank = np.full((H0, W0, 3), 245, dtype=np.uint8)
    with tifffile.TiffWriter(slide_file) as tif:
        tif.write(blank, subifds=5, tile=(256, 256), photometric="rgb", compression="deflate")
        for i in range(1, 6):
            tif.write(blank[:: 2 ** i, :: 2 ** i], subfiletype=1, tile=(256, 256), photometric="rgb",
                      compression="deflate")
    row = dict(ROW, slide="normal_901", label="normal")
    assert fetch_slide.process(row) == 0
    assert not fetch_slide.cache_path("normal_901").exists() and fetch_slide.is_cached("normal_901")


def test_download_resumes_and_checks(project, monkeypatch):
    payload = bytes(range(256)) * 4000
    calls = []

    class Reply:
        def __init__(self, start): self.data, self.status = payload[start:], 206
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, n):
            out, self.data = self.data[:n], self.data[n:]
            return out

    def fake_urlopen(req, timeout):
        start = int(req.get_header("Range").split("=")[1].rstrip("-"))
        calls.append(start)
        return Reply(start)

    monkeypatch.setattr(fetch_slide.urllib.request, "urlopen", fake_urlopen)
    dest = config.SLIDES / "x.tif"
    (config.SLIDES / "x.tif.part").write_bytes(payload[:1000])      # an interrupted earlier attempt
    import hashlib
    fetch_slide.download("http://example/x.tif", dest, len(payload), hashlib.md5(payload).hexdigest())
    assert calls == [1000] and dest.read_bytes() == payload

    (config.SLIDES / "y.tif.part").write_bytes(payload)
    with pytest.raises(IOError, match="checksum"):
        fetch_slide.download("http://example/y.tif", config.SLIDES / "y.tif", len(payload), "0" * 32)
    assert not (config.SLIDES / "y.tif.part").exists()
