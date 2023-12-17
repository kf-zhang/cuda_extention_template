from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

cxx_compiler_flags = []

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")

setup(
    name="Add",
    ext_modules=[
        CUDAExtension(
            name="Add._C",
            sources=[
                "src/bind.cpp",
                "src/Add.cpp",
                "src/Add_kernel.cu",
            ],
            extra_compile_args={"nvcc": [], "cxx": cxx_compiler_flags})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)