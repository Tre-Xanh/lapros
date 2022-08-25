all: preview

preview:
	quarto preview

publish:
	quarto publish gh-pages

pdf:
	quarto render

pluto:
	julia -e "using Pluto; Pluto.run()"
