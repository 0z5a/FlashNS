"""Check the manuscript evidence index against the files in this checkout.

Verifies every pinned result report, the LaTeX source and the PDF listed in
research/manuscript/evidence-index.json. Uses only the standard library.
Exit status 1 lists every mismatch or missing file.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path,
                        default=Path(__file__).resolve().parent.parent
                        / "research/manuscript/evidence-index.json")
    args = parser.parse_args()
    root = args.index.parent
    index = json.loads(args.index.read_text())
    expected = dict(index["evidence_sha256"])
    expected["flashns.tex"] = index["manuscript_sha256"]
    expected["flashns.pdf"] = index["pdf_sha256"]
    failures = []
    for relative, digest in sorted(expected.items()):
        path = (root / relative).resolve()
        if not path.is_file():
            failures.append(f"missing: {relative}")
        elif sha256(path) != digest:
            failures.append(f"sha256 mismatch: {relative}")
    for failure in failures:
        print(failure)
    print(f"checked {len(expected)} pinned files, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
