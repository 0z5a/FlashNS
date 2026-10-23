"""Publish the analytic vs torch.func paired cohort record from its sealed per-seed archives.

    PYTHONPATH=src python3 tools/c36_pair_func_publish.py --root ROOT --output OUT.json

tools/c36_pair_publish.py (unchanged, pinned by the evidence index) with the torch.func cohort's
protocol file and decision module: the registered decision is box376.decide_c36_pair_func over the
audit entries exactly as recorded. Everything else, including the per-arm fields and the physical
observation reader, is the pair tool's code.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c36_pair_publish as pair  # noqa: E402
from box376 import decide_c36_pair_func  # noqa: E402

if __name__ == '__main__':
    pair.PROTOCOL = 'protocol-formal-pair-func-20261022.json'
    pair.decide, pair.pair_entry = decide_c36_pair_func.decide, decide_c36_pair_func.pair_entry
    pair.main()
