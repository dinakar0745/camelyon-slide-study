import numpy as np

import tiling

XML = """<?xml version="1.0"?>
<ASAP_Annotations><Annotations>
  <Annotation Name="a" Type="Polygon" PartOfGroup="Tumor" Color="#F4FA58"><Coordinates>
    <Coordinate Order="1" X="640" Y="0"/><Coordinate Order="0" X="0" Y="0"/>
    <Coordinate Order="2" X="640" Y="640"/><Coordinate Order="3" X="0" Y="640"/>
  </Coordinates></Annotation>
  <Annotation Name="b" Type="Polygon" PartOfGroup="_2" Color="#F4FA58"><Coordinates>
    <Coordinate Order="0" X="0" Y="0"/><Coordinate Order="1" X="320" Y="0"/>
    <Coordinate Order="2" X="320" Y="320"/><Coordinate Order="3" X="0" Y="320"/>
  </Coordinates></Annotation>
</Annotations></ASAP_Annotations>"""


def test_tissue_mask_keeps_stain_and_drops_white_and_black():
    img = np.full((60, 90, 3), 245, dtype=np.uint8)   # glass
    img[:, 60:] = 0                                    # scanner padding
    img[10:50, 10:40] = (200, 90, 160)                 # pink tissue
    mask = tiling.tissue_mask(img)
    assert mask[10:50, 10:40].all()
    assert mask.sum() == 40 * 30


def test_tissue_mask_of_blank_slide_is_empty():
    assert not tiling.tissue_mask(np.full((20, 20, 3), 240, dtype=np.uint8)).any()


def test_annotation_orders_points_and_separates_exclusions():
    lesions, exclusions = tiling.parse_annotation(XML)
    assert len(lesions) == 1 and len(exclusions) == 1
    assert lesions[0].tolist() == [[0, 0], [640, 0], [640, 640], [0, 640]]


def test_lesion_mask_subtracts_exclusion():
    lesions, exclusions = tiling.parse_annotation(XML)
    mask = tiling.lesion_mask((40, 40), lesions, exclusions, mask_ds=32)
    # 640 px square at 1/32 is 20 x 20 mask pixels; the 320 px exclusion removes 10 x 10.
    assert abs(int(mask.sum()) - (400 - 100)) <= 40   # polygon edges may gain or lose a row
    assert not mask[2:8, 2:8].any() and mask[12:18, 12:18].all() and not mask[25:, 25:].any()


def test_grid_fractions_values_order_and_edge_handling():
    mask = np.zeros((10, 15), dtype=bool)   # 320 x 480 level-0 px at mask_ds 32
    mask[0:4, 0:4] = True                    # exactly the first 128 px tile
    mask[4:8, 4:6] = True                    # half of the tile at (128, 128)
    coords, frac = tiling.grid_fractions(mask, width0=480 + 50, height0=320, tile0=128, mask_ds=32)
    assert len(coords) == 4 * 2              # 530 // 128 = 4 columns, 320 // 128 = 2 rows
    assert coords[:5].tolist() == [[0, 0], [128, 0], [256, 0], [384, 0], [0, 128]]
    got = dict(zip(map(tuple, coords.tolist()), frac.tolist()))
    assert got[(0, 0)] == 1.0 and got[(128, 128)] == 0.5 and got[(256, 0)] == 0.0


def test_grid_fractions_handles_non_integer_tile_to_mask_ratio():
    mask = np.ones((9, 9), dtype=bool)
    coords, frac = tiling.grid_fractions(mask, width0=288, height0=288, tile0=100, mask_ds=32)
    assert len(coords) == 4 and np.allclose(frac, 1.0)
