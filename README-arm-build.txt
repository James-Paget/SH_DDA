  539  clang++ -std=c++14 -O3 -Wall -Wextra -pedantic -c -fPIC Beams.cpp -o Beams.o
  540  clang++ -O3 -shared Beams.o -o Beams.dylib
  541  python Beam_Testing_cpp.py

g++-14 also works, similar speed, but need to ensure the abs are changed to fabs.

g++-14 -std=c++14 -O3 -Wall -Wextra -pedantic -c -fPIC Beams.cpp -o Beams.o
g++-14 -O3 -shared Beams.o -o Beams.dylib
python Beam_Testing_cpp.py

Parallel version:

g++-14 -std=c++14 -O3 -fopenmp -Wall -Wextra -pedantic -c -fPIC Beams.cpp -o Beams.o
g++-14 -O3 -shared -fopenmp Beams.o -o Beams.dylib
python Beam_Testing_cpp.py
