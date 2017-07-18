# bare-bone makefile for/with latexmk

# name of the main .tex file
MAIN = main

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex
	latexmk -pv -pdf -pdflatex="pdflatex" $(MAIN).tex

clean:
	latexmk -CA

.PHONY: $(MAIN).pdf all clean
