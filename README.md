# SH_DDA

* Dr Simon Hanna's DDA simulation software.
* This has been tested on Mac and Linux Mint.
* This program requires g++14.
* The files here are not compiled, but can be with the following commands on Linux or Mac.

## Linux commands

```
g++-14 -std=c++14 -O3 -fopenmp -Wall -Wextra -pedantic -c -fPIC Dipoles.cpp -o Dipoles.o
g++-14 -O3 -fopenmp -fPIC -c Beams.cpp -o Beams.o
g++-14 -O3 -fopenmp -shared Dipoles.o Beams.o -o libDipoles.so
g++-14 -O3 -shared -o libBeams.so -fPIC -fopenmp Beams.cpp
```

## Mac commands

```
g++-14 -O3 -fopenmp -std=c++14 -Wall -Wextra -pedantic -c -fPIC Dipoles.cpp -o Dipoles.o
g++-14 -O3 -fopenmp -std=c++14 -Wall -Wextra -pedantic -c -fPIC Beams.cpp -o Beams.o
g++-14 -O3 -fopenmp -shared Dipoles.o Beams.o -o Dipoles.dylib
g++-14 -O3 -shared -o libBeams.dylib -fPIC -fopenmp Beams.cpp
```
