import pytest

import budget
import config


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    for name in ("SLIDES", "CACHE", "TILES", "EMBEDDINGS", "ANNOTATIONS"):
        monkeypatch.setattr(config, name, tmp_path / "data" / name.lower())
    monkeypatch.setattr(config, "DISK_FLOOR_GB", 0)
    return tmp_path


def test_counts_nested_files(project):
    (project / "data" / "a").mkdir(parents=True)
    (project / "data" / "a" / "x.bin").write_bytes(b"0" * 1000)
    (project / "y.bin").write_bytes(b"0" * 500)
    assert budget.used_bytes() == 1500


def test_reserve_allows_up_to_the_cap_and_no_further(project, monkeypatch):
    monkeypatch.setattr(config, "BUDGET_GB", 3000 / budget.GB)  # a 3000-byte budget
    (project / "x.bin").write_bytes(b"0" * 1000)
    budget.reserve(2000, "fits exactly")
    with pytest.raises(budget.BudgetError, match="budget"):
        budget.reserve(2001, "one byte too many")


def test_reserve_respects_the_disk_floor(project, monkeypatch):
    monkeypatch.setattr(config, "DISK_FLOOR_GB", 10 ** 9)  # more than any drive has free
    with pytest.raises(budget.BudgetError, match="free"):
        budget.reserve(1, "anything")


def test_missing_folder_is_zero(project):
    assert budget.folder_bytes(project / "nope") == 0
