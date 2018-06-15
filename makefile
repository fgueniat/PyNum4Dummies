all: README.md

README.md: README.md.raw
	rm -r svgs/
	rm -r README.md
	python -m readme2tex --output README.md README.md.raw
	git add .
	git commit -am 'improved README.md'
