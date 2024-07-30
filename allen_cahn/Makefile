BUILD_DIR := build

HOST_COMP  := g++-10
HOST_FLAGS := -std=c++17 -g -ggdb -Wall -Wformat -Wfloat-conversion -Wlogical-op -Wsign-conversion -Wno-unknown-pragmas -Wno-unused-function -Wno-unused-local-typedefs -Wno-missing-braces
HOST_LINK  := -lm -rdynamic -lcuda -lcudart -lglfw -lGL -lnetcdf

DEVICE_COMP  := nvcc
#Compiler options passed to the host options while compiling non-krnels inside .cu files. 
# They all have to be prefixed like so: --compiler-options -Wall --compiler-options -Wformat ... 
DEVICE_HOST_COMP_OPTIONS := -Wall -Wformat -Wfloat-conversion -Wlogical-op -Wsign-conversion -Wno-unknown-pragmas -Wno-unused-function -Wno-unused-local-typedefs -Wno-missing-field-initializers
DEVICE_FLAGS := -Xcudafe --display_error_number --extended-lambda --expt-relaxed-constexpr -DNDEBUG -std=c++17 --use_fast_math -ccbin /usr/bin/g++-10 -O3 $(foreach option, $(DEVICE_HOST_COMP_OPTIONS), --compiler-options $(option))
DEVICE_LINK  := -g -G -dlink
# DEVICE_FLAGS := -g -G -Xcudafe --display_error_number --extended-lambda --expt-relaxed-constexpr -std=c++17 --use_fast_math -ccbin /usr/bin/g++-10 $(foreach option, $(DEVICE_HOST_COMP_OPTIONS), --compiler-options $(option))

ALL_SOURCES = $(shell find -L -regex '.*/.*\.\(c\|h\|cpp\|hpp\|cu\|cuh\)$ ')
DEPENDENCIES = $(ALL_SOURCES) Makefile
D := $(BUILD_DIR)

#We do complete rebuild when anything changes. Its simpler that way.
$(D)/main.out: $(D)/simulation.o $(DEPENDENCIES) 
	$(HOST_COMP) $(HOST_FLAGS) main.cpp -o $@ $(HOST_LINK) $(D)/simulation.o

$(D)/simulation.o: $(DEPENDENCIES) 
	$(DEVICE_COMP) $(DEVICE_FLAGS) -c simulation.cu -o $@

$(D)/cojugate_gradient.out: cojugate_gradient.c
	$(HOST_COMP) $(HOST_FLAGS) cojugate_gradient.c -o $@

scratch: $(D)/cojugate_gradient.out

clean:
	rm -f $(D)/*.o

$(info $(shell mkdir -p $(D)))