all:
	julia -e 'using Weave; weave("denoise.jmd")'
