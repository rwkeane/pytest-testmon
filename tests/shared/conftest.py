import shutil
import sys

import pytest

from tests.shared.helpers import GitRepo, init_db

if shutil.which("git") is None:  # pragma: no cover
    pytest.skip("git is required", allow_module_level=True)

linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="the shared database targets Linux"
)


@pytest.fixture
def shared_db(tmp_path):
    """The machine-wide database, outside every checkout (created as by testmon-admin init)."""
    path = tmp_path / "cache" / "testmon" / "shared.sqlite"
    init_db(path)
    return path


@pytest.fixture
def repo(tmp_path):
    return GitRepo.init(tmp_path / "work" / "main-checkout")
