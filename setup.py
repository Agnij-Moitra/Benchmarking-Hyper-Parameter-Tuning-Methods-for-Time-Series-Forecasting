from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os
import multiprocessing
from Cython.Compiler import Options

# Disable Cython docstrings to reduce size
Options.docstrings = False

# Explicitly set GCC as the compiler
os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'

# Enable parallel build
os.environ['CYTHON_BUILD_PARALLEL'] = '1'

# Detect available cores
num_cores = multiprocessing.cpu_count()

extensions = [
    Extension(
        "utils",
        ["utils.pyx"],
        extra_compile_args=[
            '-O3',
            '-march=native',
            f'-flto={num_cores}'
        ],
        extra_link_args=[
            f'-flto={num_cores}'
        ],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        annotate=True,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'nonecheck': False,
            'initializedcheck': False,
            'cdivision': True,
            'overflowcheck': False,
            'infer_types': True,
            'autotestdict': False,
            'optimize.use_switch': True,
            'optimize.unpack_method_calls': True,
            'binding': False,
            'profile': False,
            'linetrace': False,
            'embedsignature': False
        },
        build_dir="build",
        cache=True,
        nthreads=15000,
    ),
    install_requires=["numpy", "pandas", "scikit-learn", "scipy", "cython"],
    python_requires=">=3.6",
    zip_safe=False,
)
