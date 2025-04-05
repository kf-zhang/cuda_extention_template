from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='custom_ops',
    ext_modules=[
        CUDAExtension(
            name='custom_ops',
            sources=['Add.cu'],
            extra_compile_args={'nvcc': ['-O2']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)