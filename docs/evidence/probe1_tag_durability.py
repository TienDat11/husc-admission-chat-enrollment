import lancedb, tempfile, shutil, os, traceback
from datetime import timedelta
os.makedirs("D:/tmp", exist_ok=True)
d = tempfile.mkdtemp(dir="D:/tmp")
results = {}
try:
    db = lancedb.connect(d)
    t = db.create_table("probe", data=[{"id": "a", "v": [0.1] * 4}])
    v_a = t.version
    t.tags.create("rel-A", v_a)
    t.add([{"id": "b", "v": [0.2] * 4}])
    v_b = t.version
    results["version_after_tag"] = v_a
    results["version_after_add"] = v_b
    results["rows_latest"] = t.count_rows()
    try:
        t.optimize(cleanup_older_than=timedelta(seconds=0), delete_unverified=False)
        results["optimize_ok"] = True
        t.checkout("rel-A")
        results["rows_at_tag_after_optimize"] = t.count_rows()
        results["TAG_SURVIVES_OPTIMIZE"] = (t.count_rows() == 1)
        t.checkout_latest()
        results["rows_back_to_latest"] = t.count_rows()
    except Exception as e:
        results["optimize_ok"] = "FAIL: " + repr(e)
        results["TAG_SURVIVES_OPTIMIZE"] = None
        results["rows_at_tag_after_optimize"] = None
        results["rows_back_to_latest"] = None
    # Test restore independently (no optimize needed)
    try:
        t.restore("rel-A")
        results["rows_after_restore"] = t.count_rows()
        results["RESTORE_BY_TAG_STRING"] = (t.count_rows() == 1)
        results["version_after_restore"] = t.version
    except Exception as e:
        results["RESTORE_BY_TAG_STRING"] = "FAIL: " + repr(e)
        results["rows_after_restore"] = None
        results["version_after_restore"] = None
    t.tags.create("prod", v_a)
    try:
        t.tags.update("prod", v_b)
        results["tags_update_ok"] = True
    except Exception as e:
        results["tags_update_ok"] = "FAIL: " + repr(e)
    results["tags_list"] = str(t.tags.list())
except Exception:
    results["EXCEPTION"] = traceback.format_exc()
finally:
    for k, v in results.items():
        print(k, "=", v)
    shutil.rmtree(d, ignore_errors=True)
