import lancedb, pyarrow as pa, tempfile, shutil, os, traceback, random
os.makedirs("D:/tmp", exist_ok=True)
d = tempfile.mkdtemp(dir="D:/tmp")
out = []
try:
    DIM = 1024
    schema = pa.schema([
        pa.field("id", pa.string(), nullable=False),
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("vector", pa.list_(pa.float32(), DIM), nullable=False),
        pa.field("data_year", pa.int32(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("effective_from", pa.int32(), nullable=False),
        pa.field("effective_to", pa.int32(), nullable=False),
        pa.field("is_tombstone", pa.bool_(), nullable=False),
    ])
    random.seed(42)
    rows = [{
        "id": "c%d" % i, "doc_id": "d%d" % (i % 20),
        "vector": [random.random() for _ in range(DIM)],
        "data_year": 2024 + (i % 3),
        "status": ["current", "superseded", "reference", "repealed"][i % 4],
        "effective_from": 20240101 + i,
        "effective_to": 99991231,
        "is_tombstone": (i % 50 == 0),
    } for i in range(2000)]
    db = lancedb.connect(d)
    t = db.create_table("probe", data=rows, schema=schema)
    CLAUSE = ("(status = 'current' AND effective_from <= 20260726 "
              "AND effective_to >= 20260726 AND is_tombstone = false)")
    q = [random.random() for _ in range(DIM)]
    out.append(("PLAN_BEFORE_INDEX", t.search(q).where(CLAUSE, prefilter=True).limit(10).explain_plan(True)))
    for col, kind in [("id","BTREE"),("doc_id","BTREE"),("data_year","BTREE"),
                      ("effective_from","BTREE"),("effective_to","BTREE"),
                      ("status","BITMAP"),("is_tombstone","BITMAP")]:
        try:
            t.create_scalar_index(col, index_type=kind)
            out.append(("index_" + col, "OK " + kind))
        except Exception as e:
            out.append(("index_" + col, "FAIL " + kind + ": " + repr(e)))
    out.append(("list_indices", str(t.list_indices())))
    out.append(("PLAN_AFTER_INDEX", t.search(q).where(CLAUSE, prefilter=True).limit(10).explain_plan(True)))
    try:
        out.append(("PLAN_SCALAR_ONLY", t.search().where(CLAUSE).limit(10).explain_plan(True)))
    except Exception as e:
        out.append(("PLAN_SCALAR_ONLY", "ERR " + repr(e)))
    try:
        nschema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 8)),
            pa.field("metadata", pa.struct([pa.field("data_year", pa.int32())])),
        ])
        nrows = [{"id": "n%d" % i, "vector": [0.1]*8, "metadata": {"data_year": 2024 + i % 3}} for i in range(100)]
        nt = db.create_table("nested", data=nrows, schema=nschema)
        out.append(("NESTED_DOTTED_FILTER_ROWS", len(nt.search().where("metadata.data_year = 2026").limit(5).to_list())))
        try:
            r2 = nt.search().where("data_year = 2026").limit(5).to_list()
            out.append(("NESTED_BARE_COL_FILTER", "returned %d rows, NO ERROR (silent swallow)" % len(r2)))
        except Exception as e:
            out.append(("NESTED_BARE_COL_FILTER", "RAISES: " + repr(e)))
    except Exception:
        out.append(("NESTED_PROBE", "ERR " + traceback.format_exc()))
except Exception:
    out.append(("EXCEPTION", traceback.format_exc()))
finally:
    for k, v in out:
        print("\n===== " + k + " =====\n" + str(v))
    shutil.rmtree(d, ignore_errors=True)
