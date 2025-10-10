#
# basic makefile 
#	make pdf:	will create a pdf file
#	make clean:	will remove all generated files
#			inclding the pdf file
#

LATEX=latex		# read the source
PDFLATEX=pdflatex
BIBTEX=bibtex
FILE=capstone

pdf:
	$(PDFLATEX) $(FILE)
	# $(LATEX) $(FILE)
	$(BIBTEX) $(FILE)
	# $(LATEX) $(FILE)
	$(PDFLATEX) $(FILE)
#	open $(FILE).pdf # this might not work for you

clean:
	rm -f capstone.blg capstone.bbl capstone.pdf capstone.log capstone.aux 
