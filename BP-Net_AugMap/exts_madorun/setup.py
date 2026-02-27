from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Blackwell GPU (Compute Capability 12.0) 対応
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6;8.9;9.0;12.0'

nvcc_args = [
    '-O3',
    '--expt-relaxed-constexpr',
    '--use_fast_math',
]

setup(
    name='BpOps',
    ext_modules=[
        CUDAExtension('BpOps',
                      [
                          'bp_cuda.cpp',
                          'bp_cuda_kernel.cu',
                      ],
                      extra_compile_args={'cxx': ['-g', '-std=c++17', '-O3'],
                                          'nvcc': nvcc_args}
                      ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })