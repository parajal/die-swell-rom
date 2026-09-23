include Mdefs.mk

.SUFFIXES:

.SUFFIXES: .o .f90 .F90 .f

.f90.o:
	$(FC) -c $(FFLAGS) $(FMODINC) $<

SRC = *.f90

OBJS = find_normals.o update_mesh_nodes_bc.o

MAINP = extrudate_swell2d_c extrudate_swell2d_c_test extrudate_swell2d_conv extrudate_test

all: objects programs tags

objects: $(OBJS)

programs: $(MAINP)

tags: $(SRC)
	ctags --fortran-kinds=+i $(SRC) $(EXTRATAGSSRC)

extrudate_swell2d_c.o: update_mesh_nodes_bc.o
extrudate_swell2d_c: $(OBJS) extrudate_swell2d_c.o
	$(FC) $(FFLAGS) -o $@ extrudate_swell2d_c.o $(OBJS) $(LIBS)

extrudate_swell2d_c_test.o: update_mesh_nodes_bc.o
extrudate_swell2d_c_test: $(OBJS) extrudate_swell2d_c_test.o
	$(FC) $(FFLAGS) -o $@ extrudate_swell2d_c_test.o $(OBJS) $(LIBS)

extrudate_swell2d_conv.o: update_mesh_nodes_bc.o
extrudate_swell2d_conv: $(OBJS) extrudate_swell2d_conv.o
	$(FC) $(FFLAGS) -o $@ extrudate_swell2d_conv.o $(OBJS) $(LIBS)

extrudate_test.o: update_mesh_nodes_bc.o
extrudate_test: $(OBJS) extrudate_test.o
	$(FC) $(FFLAGS) -o $@ extrudate_test.o $(OBJS) $(LIBS)

clean:
	$(RM)  extrudate_swell2d_c extrudate_swell2d_c_test extrudate_swell2d_conv extrudate_test *.o *.mod *.bin tags

outclean:
	$(RM) $(OUTCLEAN)

allclean: clean outclean
