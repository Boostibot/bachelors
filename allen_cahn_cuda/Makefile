D := build
HOST_COMP := g++
HOST_FLAGS = -g -ggdb -Wall -Wformat -Wfloat-conversion -Wlogical-op  -Wsign-conversion -Wno-unknown-pragmas -Wno-unused-function -Wno-unused-local-typedefs -Wno-missing-braces
HOST_LINK  = -lm -rdynamic -L/lib/cuda/include -lcuda -lcudart
# -Wdouble-promotion

DEVICE_COMP := nvcc
DEVICE_FLAGS   = -Xcudafe --display_error_number -O3
DEVICE_LINK    = -dlink

ALL_SOURCES = $(shell find -L -regex '.*/.*\.\(c\|h\|cpp\|hpp\|cu\|cuh\)$ ')
DEPENDENCIES = $(ALL_SOURCES) Makefile

#We do complete rebuild when anything changes. Its simpler that way.
$(D)/main.out: $(D)/kernel.o all
	$(HOST_COMP) main.c -o $@ $(HOST_LINK) $(D)/kernel.o

$(D)/kernel.o: 		all
	$(DEVICE_COMP) $(DEVICE_FLAGS) -c kernel.cu -o $@

all: $(DEPENDENCIES) 

clean:
	rm -f *.o

# $(info $(shell rm -rf $(D)))
$(info $(shell mkdir -p $(D)))
