include Mdefs.mk

.SUFFIXES:

.SUFFIXES: .o .f90 .F90 .f

.f90.o:
	$(FC) -c $(FFLAGS) $(FMODINC) $<

SRC = *.f90

OBJS = update_mesh_nodes_bc.o

MAINP = extrudate_test

all: objects programs tags

objects: $(OBJS)

programs: $(MAINP)

tags: $(SRC)
	ctags --fortran-kinds=+i $(SRC) $(EXTRATAGSSRC)

extrudate_test.o: update_mesh_nodes_bc.o
extrudate_test: $(OBJS) extrudate_test.o
	$(FC) $(FFLAGS) -o $@ extrudate_test.o $(OBJS) $(LIBS)

clean:
	$(RM)  extrudate_test *.o *.mod *.bin tags

outclean:
	$(RM) $(OUTCLEAN)

allclean: clean outclean
