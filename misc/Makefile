INC=.
LIB=.

# Get INC and LIB from Make.user
ifeq (exists, $(shell [ -e ./Make.user ] && echo exists ))
include ./Make.user
endif

test-salt2model : test_salt2model.o
	g++ -DUSELAPACK -DgFortran -g -Wall -O3 -Wl,-rpath,$(LIB) -L$(LIB) -o test-salt2model test_salt2model.o -lsnfit -llapack -lblas -llapack -lgfortran

test_salt2model.o : test_salt2model.cc
	g++ -DHAVE_CONFIG_H -I$(INC) -g -Wall -O3 -DUSELAPACK -DgFortran -g -Wall -O3 -MT test_salt2model.o -MD -MP -c -o test_salt2model.o test_salt2model.cc

.PHONY: clean

clean :
	rm test-salt2model
	rm test_salt2model.o
	rm test_salt2model.d

