.PHONY: docs

test:
	pytest --cov=pystrel
	mypy
	pylint pystrel/ tests/
	black --check .

docs:
	pdoc --math --docformat=numpy pystrel/ -o site/
	jupyter nbconvert --to html examples/*.ipynb
	cp examples/*.html site/

format: 
	black .