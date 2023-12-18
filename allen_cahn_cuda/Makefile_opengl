DIRS=build
CC=gcc
FLAGS=-g -ggdb -Wall -Wformat -Wfloat-conversion -Wlogical-op  -Wsign-conversion -Wno-unknown-pragmas -Wno-unused-function -Wno-unused-local-typedefs -Wno-missing-braces
LINK_FLAGS=-lm -lglfw -rdynamic

ALL_SOURCES=$(shell find -L -regex '.*/.*\.\(c\|h\|cpp\|hpp\|cu\|cuh\)$ ')
DEPENDENCIES=$(ALL_SOURCES) Makefile

# -Wdouble-promotion
build/main.out: $(DEPENDENCIES)
	$(CC) $(FLAGS) main.c -o build/main.out $(LINK_FLAGS)

clean:
	rm -rf build

# $(info $(shell rm -rf $(DIRS)))
$(info $(shell mkdir -p $(DIRS)))

