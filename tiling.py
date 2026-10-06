"""Tissue detection, tile grid and lesion-annotation rasterising. No slide I/O here.

Coordinates are level-0 pixels throughout. Masks are low-resolution arrays whose pixels
each cover `mask_ds` x `mask_ds` level-0 pixels.
"""
import xml.etree.ElementTree as ET

import numpy as np
from skimage.color import rgb2hsv
from skimage.draw import polygon as fill_polygon
from skimage.filters import threshold_otsu

# ASAP group names used in CAMELYON16: lesions are "Tumor" (older files: "_0", "_1"),
# holes cut out of a lesion are "Exclusion" (older files: "_2").
EXCLUSION_GROUPS = {"exclusion", "_2"}


def tissue_mask(rgb: np.ndarray) -> np.ndarray:
    """Boolean tissue mask from a low-resolution RGB image.

    Tissue is stained, so it is saturated; glass (white) and scanner padding (black) are not.
    The threshold is Otsu's on saturation, with a floor so an empty slide yields no tissue.
    """
    sat = rgb2hsv(rgb)[..., 1]
    thr = threshold_otsu(sat) if sat.min() < sat.max() else 1.0
    return sat > max(thr, 0.05)


def parse_annotation(xml_text: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """ASAP XML -> (lesion polygons, exclusion polygons), each an [n, 2] array of (x, y)."""
    lesions, exclusions = [], []
    for ann in ET.fromstring(xml_text).iter("Annotation"):
        pts = [(int(c.get("Order")), float(c.get("X")), float(c.get("Y"))) for c in ann.iter("Coordinate")]
        if len(pts) < 3:
            continue
        poly = np.array([(x, y) for _, x, y in sorted(pts)])
        group = (ann.get("PartOfGroup") or "").lower()
        (exclusions if group in EXCLUSION_GROUPS else lesions).append(poly)
    return lesions, exclusions


def lesion_mask(shape: tuple[int, int], lesions, exclusions, mask_ds: float) -> np.ndarray:
    """Rasterise polygons onto a (rows, cols) boolean mask: lesions filled, exclusions removed."""
    mask = np.zeros(shape, dtype=bool)
    for polys, value in ((lesions, True), (exclusions, False)):
        for p in polys:
            rr, cc = fill_polygon(p[:, 1] / mask_ds, p[:, 0] / mask_ds, shape)
            mask[rr, cc] = value
    return mask


def grid_fractions(mask: np.ndarray, width0: int, height0: int, tile0: int, mask_ds: float):
    """Mean of `mask` over every cell of a non-overlapping tile grid anchored at (0, 0).

    Returns (coords, fractions): coords is [n, 2] level-0 (x, y) of each tile's top-left
    corner, row by row; fractions is the share of each tile covered by the mask. Tiles that
    would run past the slide edge are left out.
    """
    nx, ny = width0 // tile0, height0 // tile0
    integral = np.pad(mask.astype(np.float64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    edges_x = np.clip(np.round(np.arange(nx + 1) * tile0 / mask_ds).astype(int), 0, mask.shape[1])
    edges_y = np.clip(np.round(np.arange(ny + 1) * tile0 / mask_ds).astype(int), 0, mask.shape[0])
    y0, y1, x0, x1 = edges_y[:-1, None], edges_y[1:, None], edges_x[None, :-1], edges_x[None, 1:]
    area = np.maximum((y1 - y0) * (x1 - x0), 1)
    frac = (integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]) / area
    xs, ys = np.meshgrid(np.arange(nx) * tile0, np.arange(ny) * tile0)
    coords = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.int32)
    return coords, frac.ravel().astype(np.float32)
