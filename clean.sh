#!/bin/bash

rm `find ./* -name "*.dot"`
rm `find ./* -name Makefile`
rm -r `find ./* -name CMakeFiles`
rm `find ./* -name cmake_install.cmake`
rm `find ./* -name CMakeCache.txt`
