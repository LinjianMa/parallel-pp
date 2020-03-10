include config.mk

BDIR=$(shell pwd)
ODIR=$(BDIR)/obj
SDIR=$(BDIR)/src
TDIR=$(BDIR)/tests

FCXX=$(CXX) $(CXXFLAGS)

all: test run

run: run.cxx $(ODIR)/common.o Makefile config.mk
	$(FCXX) $< $(ODIR)/common.o -o $@ $(INCLUDE_PATH) $(LIB_PATH) $(LIBS)

test: $(TDIR)/test_decomposition.cxx $(ODIR)/common.o Makefile config.mk
	$(FCXX) $< $(ODIR)/common.o -o $@ $(INCLUDE_PATH) $(LIB_PATH) $(LIBS)

$(ODIR)/common.o: common.cxx common.h config.mk
	$(FCXX) -c $< -o $@ $(INCLUDE_PATH) 

clean:
	rm -f $(ODIR)/*.o test run

format:
	clang-format -i *.cxx *.h $(SDIR)/*.cxx $(SDIR)/*.h $(SDIR)/optimizer/*.cxx $(SDIR)/optimizer/*.h
