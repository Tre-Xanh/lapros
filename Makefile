all: bookdown

bookdown:
	Rscript -e "bookdown::render_book('index.Rmd', 'all')"

pluto:
	julia -e "using Pluto; Pluto.run()"
