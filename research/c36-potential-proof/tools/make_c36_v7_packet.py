"""Build the frozen C36 v7 packet from an extracted copy of this directory and the v5 run outputs.

    python3 tools/make_c36_v7_packet.py --source EXTRACTED_C36_DIR --v5-runs V5_RUNS --packet PACKET

Copies every tracked Python and shell source under src/, the unchanged eta = 1/32 inputs (data/),
the v7 protocol, and the two v5 analytic final checkpoints named by the protocol's starts, whose
SHA-256 must match the protocol, then writes manifest.json with every file's SHA-256. The source
is a git archive of the pre-registration commit. Standard library only; refuses an existing packet.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

PROTOCOL = 'protocol-v7-ssbroyden-development-20261019.json'
DATA = ('training-source.npz', 'training-source.json', 'calibration.npz', 'calibration.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--v5-runs', type=Path, required=True)
    parser.add_argument('--packet', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.source / PROTOCOL).read_text())
    args.packet.mkdir(parents=True, exist_ok=False)
    files = {}
    for path in sorted((args.source / 'src').rglob('*')):
        if path.is_file() and path.suffix in ('.py', '.sh'):
            files['src/' + str(path.relative_to(args.source / 'src'))] = path
    for name in DATA:
        files['data/' + name] = args.source / 'data' / name
    files[PROTOCOL] = args.source / PROTOCOL
    for key, start in protocol['starts'].items():
        seed = int(key.split(':')[1])
        origin = args.v5_runs / f'seed{seed}-analytic' / 'train' / f'seed{seed}-analytic-final.pt'
        assert sha(origin) == start['checkpoint_sha256'], (key, 'start checkpoint differs from the protocol')
        files[start['packet_path']] = origin
    manifest = {'state': 'FROZEN_C36_V7_SSBROYDEN_PACKET', 'files': {}}
    for relative, path in files.items():
        target = args.packet / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        manifest['files'][relative] = {'sha256': sha(target)}
    for key, start in protocol['starts'].items():
        assert manifest['files'][start['packet_path']]['sha256'] == start['checkpoint_sha256'], key
    (args.packet / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps({'files': len(manifest['files']), 'manifest_sha256': sha(args.packet / 'manifest.json')}))


if __name__ == '__main__':
    main()
