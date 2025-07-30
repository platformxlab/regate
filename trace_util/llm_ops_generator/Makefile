PYTHON_EXEC=/usr/bin/env python3

all: cythonize

xla_hlo_parser:
	$(MAKE) -C ../xla_hlo_parser

cythonize: cython_setup.py xla_hlo_parser
	$(PYTHON_EXEC) cython_setup.py build_ext --inplace

clean:
	rm -rf *.c *.so
	rm -rf build
	rm -rf __pycache__
	$(MAKE) -C ../xla_hlo_parser clean

.PHONY: clean cythonize xla_hlo_parser all
