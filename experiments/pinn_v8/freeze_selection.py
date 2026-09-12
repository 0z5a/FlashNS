"""Freeze an explicit baseline/candidate choice from completed pilot evidence."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from runtime import frozen_protocol, source_hashes


def choice(value):
    parts = value.split(":")
    if len(parts) != 2 or parts[0] not in ("dense", "split", "compact") or parts[1] not in ("autograd", "explicit", "compiled", "cuda"):
        raise argparse.ArgumentTypeError("expected layout:seed, e.g. dense:cuda")
    return {"layout": parts[0], "seed_mode": parts[1], "activation": "cuda"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True, help="pilot suite.json")
    parser.add_argument("--baseline", type=choice, required=True)
    parser.add_argument("--candidate", type=choice, action="append", required=True)
    parser.add_argument("--rationale", required=True, help="why this is the strongest applicable baseline and this candidate is worth confirmation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite a frozen choice")
    pilot = json.loads(args.pilot.read_text())
    if not pilot.get("completed") or pilot.get("phase") != "pilot" or pilot.get("source_hashes") != source_hashes():
        raise RuntimeError("a completed pilot from this exact source is required")
    protocol_path = args.pilot.parent / "protocol.json"
    if hashlib.sha256(protocol_path.read_bytes()).hexdigest() != pilot["protocol_sha256"] or json.loads(protocol_path.read_text()) != frozen_protocol():
        raise RuntimeError("pilot protocol is not the frozen scientific protocol")
    candidates = [args.baseline, *args.candidate]
    if len({json.dumps(c, sort_keys=True) for c in candidates}) != len(candidates):
        raise ValueError("baseline and candidates must be distinct")
    for candidate in candidates:
        rows = [row for row in pilot["runs"] if all(row.get(k) == v for k, v in candidate.items())]
        if len(rows) != 2 or {row["seed"] for row in rows} != {361901, 361902}:
            raise RuntimeError(f"candidate lacks both prospective pilot seeds: {candidate}")
        if not all(row["exit_code"] == 0 and row.get("completed") and row.get("result", {}).get("converged") for row in rows):
            raise RuntimeError(f"candidate has a failed pilot observation: {candidate}")
    report = {"schema_version": 1, "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_hashes": source_hashes(), "baseline": args.baseline, "candidates": candidates,
              "rationale": args.rationale, "pilot_sha256": hashlib.sha256(args.pilot.read_bytes()).hexdigest(),
              "protocol_sha256": hashlib.sha256(json.dumps(frozen_protocol(), sort_keys=True).encode()).hexdigest(),
              "formal_seeds": list(range(75201, 75211)),
              "selection_rule": "explicit choice after pilot, before new confirmation seeds; all pilot evidence retained"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
