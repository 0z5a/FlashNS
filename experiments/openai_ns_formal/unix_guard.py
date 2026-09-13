"""Comparator's AF_UNIX restriction for a container without user systemd.

Apply libseccomp rules to this process and all descendants before exec. This
supplements, rather than substitutes for, Comparator's normal Landrun sandbox.
"""

import ctypes
import errno
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path


class ArgumentComparison(ctypes.Structure):
    _fields_ = [
        ("arg", ctypes.c_uint),
        ("op", ctypes.c_int),
        ("datum_a", ctypes.c_uint64),
        ("datum_b", ctypes.c_uint64),
    ]


def probe():
    failures = []
    for name, operation in (
        ("socket_af_unix", lambda: socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)),
        ("socketpair_af_unix", lambda: socket.socketpair(socket.AF_UNIX)),
    ):
        try:
            operation()
        except OSError as error:
            if error.errno != errno.EAFNOSUPPORT:
                failures.append(f"{name}: unexpected errno {error.errno}")
        else:
            failures.append(f"{name}: restriction ineffective")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM):
        pass
    status = Path("/proc/self/status").read_text().splitlines()
    fields = dict(line.split(":", 1) for line in status if ":" in line)
    if int(fields["CapEff"], 16) or fields["NoNewPrivs"].strip() != "1":
        failures.append("expected no effective capabilities and no_new_privs")
    if failures:
        raise RuntimeError(failures)


def main():
    if os.geteuid() == 0:
        raise RuntimeError("Comparator must run unprivileged")
    library = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int
    library.seccomp_rule_add_array.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(ArgumentComparison),
    ]
    library.seccomp_rule_add_array.restype = ctypes.c_int
    library.seccomp_load.argtypes = [ctypes.c_void_p]
    library.seccomp_load.restype = ctypes.c_int
    context = library.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    if not context:
        raise RuntimeError("seccomp_init failed")
    try:
        # Address families are int arguments; compare the low 32 bits exactly.
        rule = ArgumentComparison(0, 7, 0xFFFFFFFF, int(socket.AF_UNIX))
        for name in (b"socket", b"socketpair"):
            number = library.seccomp_syscall_resolve_name(name)
            if number < 0:
                raise RuntimeError(f"unknown syscall {name!r}")
            result = library.seccomp_rule_add_array(
                context,
                0x00050000 | errno.EAFNOSUPPORT,
                number,
                1,
                ctypes.byref(rule),
            )
            if result:
                raise RuntimeError(f"seccomp_rule_add_array: {result}")
        result = library.seccomp_load(context)
        if result:
            raise RuntimeError(f"seccomp_load: {result}")
    finally:
        library.seccomp_release(context)
    probe()
    # An exec'd child must retain the exact same restriction.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); from unix_guard import probe; probe()",
            str(Path(__file__).resolve().parent),
        ],
        check=True,
        close_fds=True,
    )
    print(
        json.dumps(
            {
                "guard": "libseccomp_deny_af_unix_socket_and_socketpair",
                "uid": os.geteuid(),
                "self_test": "passed",
                "exec_inheritance": "passed",
                "script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            }
        ),
        flush=True,
    )
    arguments = sys.argv[1:]
    if arguments == ["--self-test"]:
        return
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if not arguments:
        raise RuntimeError("provide --self-test or a command to execute")
    os.execvpe(arguments[0], arguments, os.environ)


if __name__ == "__main__":
    main()
