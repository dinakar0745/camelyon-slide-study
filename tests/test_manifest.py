import manifest

PAGE = b"""<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Name>camelyon-dataset</Name>
<NextContinuationToken>abc==</NextContinuationToken><IsTruncated>true</IsTruncated>
<Contents><Key>CAMELYON16/images/normal_001.tif</Key><Size>1000</Size></Contents>
<Contents><Key>CAMELYON16/images/tumor_001.tif</Key><Size>2000</Size></Contents>
</ListBucketResult>"""


def test_parse_listing_reads_keys_sizes_and_token():
    items, token = manifest.parse_listing(PAGE)
    assert items == [("CAMELYON16/images/normal_001.tif", 1000), ("CAMELYON16/images/tumor_001.tif", 2000)]
    assert token == "abc=="


def test_parse_listing_last_page_has_no_token():
    last = PAGE.replace(b"<NextContinuationToken>abc==</NextContinuationToken>", b"")
    assert manifest.parse_listing(last)[1] is None


def test_parse_checksums_keys_by_file_name():
    text = "d41d8cd98f00b204e9800998ecf8427e  images/normal_001.tif\nnot a checksum line\n"
    assert manifest.parse_checksums(text) == {"normal_001.tif": "d41d8cd98f00b204e9800998ecf8427e"}


def test_parse_reference_with_and_without_header():
    bare = "test_001,Tumor,IDC,Macro\ntest_002,Normal,Normal,None\n"
    headed = "image,type,subtype,class\ntest_001.tif,tumor,idc,macro\nTest_002,negative,,\n"
    for text in (bare, headed):
        ref = manifest.parse_reference(text)
        assert ref["test_001"] == ("tumor", "macro")
        assert ref["test_002"][0] == "normal"


def test_build_rows_assigns_split_and_label():
    keys = {
        "CAMELYON16/images/normal_001.tif": 10,
        "CAMELYON16/images/tumor_001.tif": 20,
        "CAMELYON16/images/test_001.tif": 30,
        "CAMELYON16/images/test_002.tif": 40,   # not in the reference -> skipped
        "CAMELYON16/masks/tumor_001_mask.tif": 5,
    }
    rows = manifest.build_rows(keys, {"tumor_001.tif": "f" * 32}, {"test_001": ("tumor", "micro")}, {"tumor_001"})
    got = {r["slide"]: (r["split"], r["label"], r["lesion"], r["bytes"], r["annotation"]) for r in rows}
    assert got == {
        "normal_001": ("train", "normal", "", 10, ""),
        "tumor_001": ("train", "tumor", "", 20, "tumor_001"),
        "test_001": ("test", "tumor", "micro", 30, ""),
    }
    assert rows[[r["slide"] for r in rows].index("tumor_001")]["md5"] == "f" * 32
