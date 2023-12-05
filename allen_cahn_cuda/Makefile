DIRS=build
CC=gcc
FLAGS=-rdynamic -g -ggdb
LINK_FLAGS=-lm -lglfw

main: main.o platform.o
	$(CC) build/main.o build/platform.o -o build/main $(LINK_FLAGS)

main.o: main.c lib/platform.h extrenal/include/glfw/glfw3.h
	$(CC) $(FLAGS) -c main.c -o build/main.o
	
platform.o: lib/platform.h lib/platform_linux.c
	$(CC) $(FLAGS) -c lib/platform_linux.c -o build/platform.o

clean:
	rm -rf build

$(info $(shell rm -rf $(DIRS)))
$(info $(shell mkdir -p $(DIRS)))

