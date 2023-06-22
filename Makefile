.PHONY: docs

test:
	pytest
	mypy
	pylint pystrel/ tests/

docs:
	pdoc --math --docformat=numpy pystrel/ -o site/
