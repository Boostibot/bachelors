DIRS=build
CC=gcc
FLAGS= -g -ggdb -Wall -Wformat -Wfloat-conversion -Wlogical-op  -Wsign-conversion -Wno-unknown-pragmas -Wno-unused-function -Wno-unused-local-typedefs -Wno-missing-braces
LINK_FLAGS=-lm -lglfw -rdynamic

ALL_SOURCES=$(shell find -L -regex '.*/.*\.\(c\|h\|cpp\|hpp\|cu\|cuh\)$ ')
DEPENDENCIES=$(ALL_SOURCES) Makefile
# @TODO: add format flags to format and logging functions
# -Wdouble-promotion

#We do complete rebuild when anything changes. Its simpler that way.
build/main.out: build/main.o build/platform.o $(DEPENDENCIES)
	$(CC) build/main.o build/platform.o -o build/main.out $(LINK_FLAGS)

build/main.o: $(DEPENDENCIES)
	$(CC) $(FLAGS) -c main.c -o build/main.o
	
build/platform.o: $(DEPENDENCIES)
	$(CC) $(FLAGS) -c lib/platform_linux.c -o build/platform.o

clean:
	rm -rf build

# $(info $(shell rm -rf $(DIRS)))
$(info $(shell mkdir -p $(DIRS)))

