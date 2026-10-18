"""Build the frozen C36 v6 packet from an extracted copy of this directory.

    python3 tools/make_c36_v6_packet.py --source EXTRACTED_C36_DIR --packet PACKET

Copies every tracked Python source under src/, the eta = 1/16 inputs (data-eta16/ becomes
data/) and the v6 protocol, then writes manifest.json with each file's SHA-256. The source is a
git archive of the pre-registration commit. Standard library only; refuses an existing packet.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

PROTOCOL = 'protocol-v6-eta16-development-20261018.json'
DATA = ('training-source.npz', 'training-source.json', 'calibration.npz', 'calibration.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--packet', type=Path, required=True)
    args = parser.parse_args()
    args.packet.mkdir(parents=True, exist_ok=False)
    files = {}
    for path in sorted((args.source / 'src').rglob('*')):
        if path.is_file() and path.suffix in ('.py', '.sh'):
            files['src/' + str(path.relative_to(args.source / 'src'))] = path
    for name in DATA:
        files['data/' + name] = args.source / 'data-eta16' / name
    files[PROTOCOL] = args.source / PROTOCOL
    manifest = {'state': 'FROZEN_C36_V6_ETA16_PACKET', 'files': {}}
    for relative, path in files.items():
        target = args.packet / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        manifest['files'][relative] = {'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
    (args.packet / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps({'files': len(manifest['files']),
                      'manifest_sha256': hashlib.sha256((args.packet / 'manifest.json').read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
