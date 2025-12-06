# Makefile for Sphinx documentation
# Berkeley SciComp Framework

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = .
BUILDDIR     = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean html latexpdf

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Berkeley-specific targets
clean:
	rm -rf $(BUILDDIR)/*
	@echo "Build directory cleaned."

html:
	@echo "Building HTML documentation with Berkeley styling..."
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS)
	@echo "HTML documentation built successfully!"
	@echo "Open $(BUILDDIR)/html/index.html in your browser."

html-fast:
	@echo "Fast HTML build (no API docs regeneration)..."
	$(SPHINXBUILD) -b html -E $(SOURCEDIR) $(BUILDDIR)/html $(SPHINXOPTS)

latexpdf:
	@echo "Building LaTeX PDF documentation..."
	$(SPHINXBUILD) -b latex $(SOURCEDIR) $(BUILDDIR)/latex $(SPHINXOPTS)
	@echo "Running LaTeX files through pdflatex..."
	$(MAKE) -C $(BUILDDIR)/latex all-pdf
	@echo "PDF documentation built successfully!"

berkeley-docs: clean html
	@echo "🐻 Berkeley SciComp documentation build complete! 🐻"
	@echo "💙💛 Go Bears! 💙💛"