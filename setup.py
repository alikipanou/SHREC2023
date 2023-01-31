from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='changenetwork',
    version='1.0.0',
    ext_modules=[
        CUDAExtension(
            name='changenetwork.ext',
            sources=[
                'changenetwork/extensions/extra/cloud/cloud.cpp',
                'changenetwork/extensions/cpu/grid_subsampling/grid_subsampling.cpp',
                'changenetwork/extensions/cpu/grid_subsampling/grid_subsampling_cpu.cpp',
                'changenetwork/extensions/cpu/radius_neighbors/radius_neighbors.cpp',
                'changenetwork/extensions/cpu/radius_neighbors/radius_neighbors_cpu.cpp',
                'changenetwork/extensions/pybind.cpp',
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)