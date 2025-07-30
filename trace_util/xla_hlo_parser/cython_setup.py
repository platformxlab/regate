##### Cython compile script for npusim

from distutils.core import setup
from setuptools import Extension
from Cython.Build import cythonize

# from Cython.Compiler.Options import get_directive_defaults
# directive_defaults = get_directive_defaults()

# directive_defaults['linetrace'] = True
# directive_defaults['binding'] = True

# , define_macros=[('CYTHON_TRACE', '1')]

setup(
    name="xla_hlo_parser_proj",
    ext_modules = cythonize([
            Extension("xla_hlo_structures", ["xla_hlo_structures.py"]),
            Extension("xla_hlo_trace_parser", ["xla_hlo_trace_parser.py"])
        ],
        language_level=3
    )
)