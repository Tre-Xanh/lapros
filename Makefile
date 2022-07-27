all: pdf

pdf: docs/_main.pdf
docs/_main.pdf: index.Rmd
	make bookdown

bookdown:
	Rscript -e "bookdown::render_book('index.Rmd', 'all')"

pluto:
	julia -e "using Pluto; Pluto.run()"

env:
	Rscript -e 'install.packages("bookdown")'
