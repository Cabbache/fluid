#!/bin/sh
#g++ -o fluid.exe fluid.cpp -lSDL2 -I/usr/include/SDL2 -D_REENTRANT -O3 -march=native -lSDL2_image -lm
#g++ -o fluid.exe fluid.cpp -lSDL2 -I/usr/include/SDL2 -D_REENTRANT -ggdb3 -O0 -Wall -lSDL2_image -lm
g++ -o fluid.exe fluid.cpp -lSDL2 -I/usr/include/SDL2 -D_REENTRANT -g -lSDL2_image -lm