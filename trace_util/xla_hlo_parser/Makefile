PYTHON_EXEC=/usr/bin/python3

xla_hlo_parser: cython_setup.py xla_hlo_trace_parser.py xla_hlo_structures.py
	$(PYTHON_EXEC) cython_setup.py build_ext --inplace

clean:
	rm -rf *.c *.so
	rm -rf build

.PHONY: clean

