all: preview

FORMAT?=html

preview:
	quarto preview --no-browser --to $(FORMAT)

publish:
	quarto publish gh-pages --no-browser

render:
	quarto render --to pdf

pluto:
	julia -e "using Pluto; Pluto.run()"
