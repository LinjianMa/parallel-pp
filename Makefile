include config.mk

BDIR=$(shell pwd)
ODIR=$(BDIR)/obj
SDIR=$(BDIR)/src
TDIR=$(BDIR)/tests
UDIR=$(SDIR)/utils

FCXX=$(CXX) $(CXXFLAGS)

all: test run

run: run.cxx $(ODIR)/common.o $(ODIR)/dimension_tree.o Makefile config.mk
	$(FCXX) $< $(ODIR)/common.o $(ODIR)/dimension_tree.o -o $@ $(INCLUDE_PATH) $(LIB_PATH) $(LIBS)

test: $(TDIR)/test_decomposition.cxx $(ODIR)/common.o $(ODIR)/dimension_tree.o Makefile config.mk
	$(FCXX) $< $(ODIR)/common.o $(ODIR)/dimension_tree.o -o $@ $(INCLUDE_PATH) $(LIB_PATH) $(LIBS)

$(ODIR)/common.o: $(UDIR)/common.cxx $(UDIR)/common.h config.mk
	$(FCXX) -c $< -o $@ $(INCLUDE_PATH)

$(ODIR)/dimension_tree.o: $(UDIR)/dimension_tree.cxx $(UDIR)/dimension_tree.h config.mk
	$(FCXX) -c $< -o $@ $(INCLUDE_PATH)

.PHONY: clean
clean:
	rm -f $(ODIR)/*.o test run

.PHONY: format
format:
	clang-format -i *.cxx $(SDIR)/*.cxx $(SDIR)/*.h $(SDIR)/utils/*.cxx $(SDIR)/utils/*.h $(SDIR)/optimizer/*.cxx $(SDIR)/optimizer/*.h $(TDIR)/*.cxx
