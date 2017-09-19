all: main.cpp
	g++ -std=c++11 -fopenmp -O2 main.cpp


clean: a.out
	rm -f ./a.out
