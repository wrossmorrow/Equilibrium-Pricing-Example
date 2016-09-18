
CC    = gcc
CFLAGS = -arch $(ARCH) -I/usr/local/include -O3 -Wall
ARCH	= x86_64
		
RANLIBS  = -lunuran -lrngstreams 
BLASLIBS = -framework Accelerate 

PATHINCDIR = /usr/local/include/path
PATHLIBDIR = /usr/local/lib/path/$(ARCH)
PATHLIBS = $(RANLIBS) $(BLASLIBS) -L$(PATHLIBDIR) -lpath47 -lpathextras -lgfortran -lm

KININC = -I/usr/local/include/sundials -I/usr/local/include/kinsol
KINLIBS = $(RANLIBS) $(BLASLIBS) -lsundials_nvecserial -lsundials_kinsol

KTRLIBS = $(RANLIBS) $(BLASLIBS) -lknitro -lktrextras

SNOPT_INCLUDE   = /usr/local/include/snopt
SNOPT_LIBDIR	= /usr/local/lib
SNOPT_LIBS  	= -lsnopt_c -lsnprint_c -lblas_c
SNOPT_AR     	= libsnopt_c libsnprint_c libblas_c libsnoextras
SNOPT_AR_LIBS 	= $(SNOPT_AR:%=$(SNOPT_LIBDIR)/%.a)

F2C_INCLUDE     = /usr/local/include/snopt
F2C_LIBDIR	= /usr/local/lib

all: path kinsol knitro snopt

zetafpi: 
	$(CC) $(CFLAGS) -o sZFPI CY2005BLP95Example-ZFPI.c $(RANLIBS) $(BLASLIBS) 

nmfpi: 
	$(CC) $(CFLAGS) -o sNMFPI CY2005BLP95Example-NMFPI.c $(RANLIBS) $(BLASLIBS) 

path: 
	$(CC) $(CFLAGS) -o sPATH CY2005BLP95Example-PATH.c -I$(PATHINCDIR) $(PATHLIBS)

trpath: 
	$(CC) $(CFLAGS) -o sTRPATH CY2005BLP95Example-PATH-TR.c -I$(PATHINCDIR) $(PATHLIBS)

kinsol: 
	$(CC) $(CFLAGS) -o sKINSOL CY2005BLP95Example-KINSOL.c $(KININC) $(KINLIBS)

knitro: 
	$(CC) $(CFLAGS) -o sKNITRO CY2005BLP95Example-KNITRO.c $(KTRLIBS)

snopt: $(F2C_INCLUDE)/f2c.h snfilewrapper.o

	$(CC) $(CFLAGS) -o sSNOPT CY2005BLP95Example-SNOPT7.c \
		$(INCLUDE) -I$(SNOPT_INCLUDE) snfilewrapper.o \
		$(RANLIBS) \
		-L/usr/local/lib $(SNOPT_AR_LIBS) $(F2C_LIBDIR)/libf2c.a $(BLASLIBS) -lm

snfilewrapper.o: snfilewrapper.c

	$(CC) -c $(CFLAGS) -I$(F2C_INCLUDE) $< -o $@

# Fake target to remind people to set the F2C environment variable with SNOPT

$(F2C_INCLUDE)/f2c.h:
	@echo "Could not find the f2c distribution."
	@echo "Set the following environment variables:"
	@echo "  F2CINCLUDE should be the path to f2c.h"
	@false