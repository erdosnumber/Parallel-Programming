CC = g++
CFLAGS = -Wall -Wextra -pedantic -fopenmp

.PHONY: all clean pthread_run

all: pthread openmp sequential

pthread: pthread.cpp
	$(CC) $(CFLAGS) -o pthread pthread.cpp -lpthread

openmp: openmp.cpp
	$(CC) $(CFLAGS) -o openmp openmp.cpp -fopenmp

sequential: sequential.cpp
	$(CC) $(CFLAGS) -o sequential sequential.cpp


clean:
	rm -f pthread openmp sequential
