BUILD_DIR := build

HOST_COMP  := g++
HOST_FLAGS := -g -ggdb -Wall -Wformat -Wfloat-conversion -Wlogical-op  -Wsign-conversion -Wno-unknown-pragmas -Wno-unused-function -Wno-unused-local-typedefs -Wno-missing-braces
HOST_LINK  := -lm -rdynamic -lcuda -lcudart -lglfw -lGL

DEVICE_COMP  := nvcc
DEVICE_FLAGS := -Xcudafe --display_error_number -O3
DEVICE_LINK  := -dlink

ALL_SOURCES = $(shell find -L -regex '.*/.*\.\(c\|h\|cpp\|hpp\|cu\|cuh\)$ ')
DEPENDENCIES = $(ALL_SOURCES) Makefile
D := $(BUILD_DIR)

#We do complete rebuild when anything changes. Its simpler that way.
$(D)/main.out: $(D)/kernel.o $(DEPENDENCIES) 
	$(HOST_COMP) $(HOST_FLAGS) main.cpp -o $@ $(HOST_LINK) $(D)/kernel.o

$(D)/kernel.o: $(DEPENDENCIES) 
	$(DEVICE_COMP) $(DEVICE_FLAGS) -c kernel.cu -o $@

clean:
	rm -f *.o

$(info $(shell mkdir -p $(D)))