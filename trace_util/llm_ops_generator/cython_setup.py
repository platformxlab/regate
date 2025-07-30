##### Cython compile script for npusim

from setuptools import Extension, setup
from Cython.Build import cythonize

# from Cython.Compiler.Options import get_directive_defaults
# directive_defaults = get_directive_defaults()

# directive_defaults['linetrace'] = True
# directive_defaults['binding'] = True

# , define_macros=[('CYTHON_TRACE', '1')]

setup(
    name="llm_ops_generator",
    ext_modules=cythonize(
        [
            Extension("llm_ops_lib", ["llm_ops_lib.py"]),
            Extension("op_analysis_lib", ["op_analysis_lib.py"]),
            Extension("power_analysis_lib", ["power_analysis_lib.py"]),
            Extension("energy_carbon_analysis_lib", ["energy_carbon_analysis_lib.py"]),
        ],
        language_level=3,
    ),
)
