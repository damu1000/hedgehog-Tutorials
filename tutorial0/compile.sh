#!/bin/bash

#update hedgehog include path
g++-8 -I$HOME/hedgehog/hedgehog/build/include/ ./tutorial0.cc -std=c++17 -pthread -o tutorial0

