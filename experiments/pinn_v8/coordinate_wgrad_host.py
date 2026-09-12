"""Compile the actual coordinate-wgrad scalar helper for independent CPU checks.

This does not compile or execute CUDA. CUDA indexing/streams require separate
GPU preflight. The host and CUDA paths share only the production scalar helper.
"""

import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
WRAPPER = r'''
#include "coordinate_wgrad.cuh"
template<int D>
void run(const double* d, const double* x, double* partials, double* dw,
         double* db, int64_t full, int64_t value, int channels) {
  const int64_t points=full+value;
  const int64_t tiles=points/flashns_coordinate_wgrad::point_tile+
                      (points%flashns_coordinate_wgrad::point_tile!=0);
  for (int64_t t=0;t<tiles;++t)
    for (int c=0;c<channels;++c)
      flashns_coordinate_wgrad::tile<D>(d,x,full,value,channels,t,c,
                                       partials+(t*channels+c)*(D+1));
  for (int c=0;c<channels;++c)
    flashns_coordinate_wgrad::finish<D>(partials,tiles,channels,c,dw,db);
}
extern "C" int host_coordinate_wgrad(int dimension, const double* d, const double* x,
    double* partials, double* dw, double* db, int64_t full, int64_t value, int channels) {
  if ((dimension!=2 && dimension!=3) || full<0 || value<0 || channels<1) return 1;
  if (dimension==2) run<2>(d,x,partials,dw,db,full,value,channels);
  else run<3>(d,x,partials,dw,db,full,value,channels);
  return 0;
}
'''


def build_host(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    compiler = shutil.which("c++")
    if compiler is None:
        raise RuntimeError("a C++17 host compiler is required")
    source, library = output/"coordinate_wgrad_host.cpp", output/"coordinate_wgrad_host.so"
    source.write_text(WRAPPER)
    command = [compiler, "-std=c++17", "-O3", "-ffp-contract=off", "-shared", "-fPIC",
               "-I"+str(HERE), str(source), "-o", str(library)]
    result = subprocess.run(command, capture_output=True, text=True)
    (output/"build.log").write_text(result.stdout+result.stderr)
    report = {"compiled": result.returncode == 0, "command": command, "exit_code": result.returncode,
              "compiler": subprocess.check_output([compiler, "--version"], text=True),
              "scope": "host C++ scalar contraction only; no GPU correctness or performance",
              "source_hashes": {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                                for name in ("coordinate_wgrad.cuh", "coordinate_wgrad_host.py")}}
    if result.returncode == 0:
        report["library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
    (output/"build.json").write_text(json.dumps(report, indent=2)+"\n")
    if result.returncode:
        raise RuntimeError((output/"build.log").read_text())
    return HostCoordinateWgrad(library)


class HostCoordinateWgrad:
    def __init__(self, library):
        self.library = ctypes.CDLL(str(library))
        self.function = self.library.host_coordinate_wgrad
        self.function.argtypes = ([ctypes.c_int]+[ctypes.c_void_p]*5
                                  + [ctypes.c_int64]*2+[ctypes.c_int])
        self.function.restype = ctypes.c_int

    def __call__(self, layout, coordinates, derivative):
        import torch
        if any(t.device.type != "cpu" or t.dtype != torch.float64 or not t.is_contiguous()
               for t in (coordinates, derivative)):
            raise ValueError("contiguous CPU FP64 inputs required")
        layout.check(derivative)
        if coordinates.shape != (layout.points, layout.dimension) or derivative.shape[1] < 1:
            raise ValueError("coordinate/derivative shape mismatch")
        channels = derivative.shape[1]
        partials = derivative.new_empty(((layout.points+7)//8, channels, layout.dimension+1))
        dw, db = derivative.new_empty((channels, layout.dimension)), derivative.new_empty((channels,))
        self.call_into(layout, coordinates, derivative, partials, dw, db)
        return dw, db

    def call_into(self, layout, coordinates, derivative, partials, dw, db):
        status = self.function(layout.dimension, derivative.data_ptr(), coordinates.data_ptr(),
                               partials.data_ptr(), dw.data_ptr(), db.data_ptr(),
                               layout.full_points, layout.value_points, derivative.shape[1])
        if status:
            raise RuntimeError(f"host contraction failed: {status}")
