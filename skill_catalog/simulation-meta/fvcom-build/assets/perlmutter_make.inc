INSTALLDIR = $(TOPDIR)/libs/install
DEF_FLAGS = -P -traditional -nostdinc
LIBDIR = -L$(INSTALLDIR)/lib
INCDIR = -I$(INSTALLDIR)/include
DTLIBS = -ljulian
PARLIB = -lmetis
MODEL_FLAGS = -DDOUBLE_PRECISION -DSINGLE_OUTPUT -DWET_DRY -DMULTIPROCESSOR -DLIMITED_NO -DGCN -DAIR_PRESSURE
CPP = /usr/bin/cpp
FC = ftn
CC = cc
CXX = CC
COPTIONS = -O2 -std=gnu89 -Wno-implicit-function-declaration -Wno-incompatible-pointer-types
CFLAGS = $(COPTIONS)
CPPFLAGS = $(DEF_FLAGS) $(COMPILER)
CPPARGS = $(CPPFLAGS) $(MODEL_FLAGS)
FFLAGS = $(OPT)
MDEPFLAGS = --cpp --fext=f90 --file=-
RANLIB = ranlib
AR = ar rc
LIBS = $(LIBDIR) $(PARLIB) $(DTLIBS)
INCS = $(INCDIR)
