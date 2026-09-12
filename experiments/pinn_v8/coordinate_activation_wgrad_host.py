"""CPU compilation of the actual G2 helper and its materialized G1 control."""

import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
WRAPPER = r'''
#include <vector>
#include "coordinate_activation_wgrad.cuh"
template<int D>
void run(int fused, const double* h, const double* aux, const double* seed, const double* x,
         double* partials, double* dw, double* db, int64_t full, int64_t value, int channels) {
  constexpr int Q=D==2 ? 10 : 20;
  const int64_t points=full+value, rows=full*Q+value;
  const int64_t tiles=points/flashns_coordinate_wgrad::point_tile+
                      (points%flashns_coordinate_wgrad::point_tile!=0);
  std::vector<double> materialized;
  if (!fused) {
    materialized.resize(rows*channels);
    for (int64_t p=0;p<points;++p) {
      const int64_t row=p<full ? p*Q : full*Q+p-full;
      for (int c=0;c<channels;++c) {
        double d[Q];
        const int64_t offset=row*channels+c;
        flashns_coordinate_wgrad::point_vjp<D>(h+offset,seed+offset,channels,aux[p*channels+c],p<full,d);
        for (int j=0;j<(p<full ? Q : 1);++j) materialized[offset+int64_t(j)*channels]=d[j];
      }
    }
  }
  for (int64_t t=0;t<tiles;++t)
    for (int c=0;c<channels;++c) {
      double* out=partials+(t*channels+c)*(D+1);
      if (fused) flashns_coordinate_wgrad::tile_vjp<D>(h,aux,seed,x,full,value,channels,t,c,out);
      else flashns_coordinate_wgrad::tile<D>(materialized.data(),x,full,value,channels,t,c,out);
    }
  for (int c=0;c<channels;++c)
    flashns_coordinate_wgrad::finish<D>(partials,tiles,channels,c,dw,db);
}
extern "C" int host_coordinate_activation_wgrad(int dimension, int fused, const double* h,
    const double* aux, const double* seed, const double* x, double* partials,
    double* dw, double* db, int64_t full, int64_t value, int channels) {
  if ((dimension!=2 && dimension!=3) || full<0 || value<0 || channels<1 || (fused!=0 && fused!=1)) return 1;
  if (dimension==2) run<2>(fused,h,aux,seed,x,partials,dw,db,full,value,channels);
  else run<3>(fused,h,aux,seed,x,partials,dw,db,full,value,channels);
  return 0;
}
'''


def build_host(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    compiler = shutil.which("c++")
    if compiler is None: raise RuntimeError("a C++17 host compiler is required")
    source, library = output/"coordinate_activation_wgrad_host.cpp", output/"coordinate_activation_wgrad_host.so"
    source.write_text(WRAPPER)
    command = [compiler, "-std=c++17", "-O3", "-ffp-contract=off", "-shared", "-fPIC",
               "-I"+str(HERE), str(source), "-o", str(library)]
    result = subprocess.run(command, capture_output=True, text=True)
    (output/"build.log").write_text(result.stdout+result.stderr)
    paths = [HERE/name for name in ("coordinate_activation_wgrad.cuh", "coordinate_wgrad.cuh", "coordinate_activation_wgrad_host.py")]
    paths.append(HERE.parent/"cuda_jet_h100/stable_jet.cuh")
    report = {"compiled": result.returncode == 0, "command": command, "exit_code": result.returncode,
              "compiler": subprocess.check_output([compiler, "--version"], text=True),
              "scope": "host full stable VJP and G1/G2 coordinate contraction; no CUDA runtime validation",
              "source_hashes": {str(path.relative_to(HERE.parents[1])): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in paths}}
    if result.returncode == 0: report["library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
    (output/"build.json").write_text(json.dumps(report, indent=2)+"\n")
    if result.returncode: raise RuntimeError((output/"build.log").read_text())
    return HostCoordinateActivationWgrad(library)


class HostCoordinateActivationWgrad:
    def __init__(self, library):
        self.library = ctypes.CDLL(str(library))
        self.function = self.library.host_coordinate_activation_wgrad
        self.function.argtypes = [ctypes.c_int]*2+[ctypes.c_void_p]*7+[ctypes.c_int64]*2+[ctypes.c_int]
        self.function.restype = ctypes.c_int

    def __call__(self, layout, coordinates, hidden, aux, seed, *, fused=True):
        import torch
        tensors = (coordinates, hidden, aux, seed)
        if any(t.device.type != "cpu" or t.dtype != torch.float64 or not t.is_contiguous() for t in tensors):
            raise ValueError("contiguous CPU FP64 inputs required")
        layout.check(hidden)
        layout.check(seed)
        channels = hidden.shape[1]
        if (coordinates.shape != (layout.points, layout.dimension) or channels<1
                or aux.shape != (layout.points, channels) or seed.shape != hidden.shape):
            raise ValueError("coordinate/H/a1/bar_H shape mismatch")
        partials = hidden.new_empty(((layout.points+7)//8, channels, layout.dimension+1))
        dw, db = hidden.new_empty((channels, layout.dimension)), hidden.new_empty((channels,))
        status = self.function(layout.dimension, int(fused), hidden.data_ptr(), aux.data_ptr(), seed.data_ptr(),
                               coordinates.data_ptr(), partials.data_ptr(), dw.data_ptr(), db.data_ptr(),
                               layout.full_points, layout.value_points, channels)
        if status: raise RuntimeError(f"host G1/G2 contraction failed: {status}")
        return dw, db
