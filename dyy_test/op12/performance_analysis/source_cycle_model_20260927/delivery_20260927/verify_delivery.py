"""Read-only verification of the independent source-model delivery."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
load = lambda name: json.loads((root / name).read_text(encoding="utf-8"))
sha = lambda file: hashlib.sha256(file.read_bytes()).hexdigest()
freeze = load("SOURCE_MODEL_FREEZE.json")
delivery = load("delivery_integrity.json")
for item in freeze["files"]:
    assert sha(root / item["path"]) == item["sha256"], item["path"]
for item in delivery["outputs"]:
    assert sha(root / item["path"]) == item["sha256"], item["path"]
for item in freeze["source_files"]:
    assert sha(root / "source_archive" / Path(item["path"]).name) == item["sha256"]
for item in load("event_sanity_check.json")["sources"]:
    assert sha(root / "event_raw" / Path(item["path"]).name) == item["sha256"]
validation = load("outputs/source_model_20260927/validation.json")
assert not validation["formula_errors"]
assert all(x["actual"] == x["expected"] for x in validation["calendar_formula_checks"])
assert not load("approved_model.json")["timing_used_for_model"]
print(json.dumps({"frozen_models": len(freeze["files"]), "source_hashes": len(freeze["source_files"]), "delivered_output_hashes": len(delivery["outputs"]), "excel_calendar_checks": len(validation["calendar_formula_checks"]), "status": "PASS"}))
