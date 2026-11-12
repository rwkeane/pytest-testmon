import pytest

from testmon.db import DB
from testmon.process_code import Module, create_fingerprint
from testmon.selection import compute_selection
from testmon.testmon_core import SourceTree

SOURCE = "def f():\n    return 1\n"
COMMENTED = "# a comment\ndef f():\n    return 1\n"
BODY_LINE = {SOURCE: 2, COMMENTED: 3}


def record(db, exec_id, test_name, source):
    module = Module(source_code=source)
    db.insert_test_file_fps(
        {
            test_name: {
                "deps": [
                    {
                        "filename": "a.py",
                        "fsha": module.fs_fsha,
                        "method_checksums": create_fingerprint(
                            module, {BODY_LINE[source]}
                        ),
                    }
                ]
            }
        },
        exec_id,
    )
    return module.fs_fsha


def memoised(db, fsha):
    return db.con.execute(
        "SELECT valid FROM fp_match WHERE fsha = ?", (fsha,)
    ).fetchall()


class TestFingerprintAfterBytesOnlyChange:
    @pytest.mark.parametrize(
        "old_source, new_source", [(SOURCE, COMMENTED), (COMMENTED, SOURCE)]
    )
    def test_rerecorded_test_stores_current_fsha(
        self, tmp_path, old_source, new_source
    ):
        datafile = str(tmp_path / ".testmondata")
        db = DB(datafile)
        exec_id, _ = db.fetch_or_create_environment("default", "", "3")
        record(db, exec_id, "test_a.py::test_a", old_source)
        record(db, exec_id, "test_b.py::test_b", new_source)

        new_fsha = record(db, exec_id, "test_a.py::test_a", new_source)

        # the current content is known to match: no reanalysis is needed for it
        assert [row[0] for row in memoised(db, new_fsha)] == [1]
        tree = SourceTree("")
        tree.cache["a.py"] = Module(source_code=new_source, fs_fsha=new_fsha)
        result = compute_selection(db, exec_id, tree, use_snapshots=False)
        assert result.deselectable == {"test_a.py::test_a", "test_b.py::test_b"}
        assert result.new_memo == []
        db.con.close()
