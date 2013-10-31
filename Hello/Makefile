all:	main

main:
	gcc -Wall -std=c99 -o main main.c

cuda:
	nvcc -arch=sm_20 -o cuda main.cu

hello:
	nvcc -arch=sm_20 -o hello hello.cu
xy:
	nvcc -arch=sm_20 -o xy xy.cu
test:
	./main ../Crimes_id-ward-year-month.csv

clean:
	rm main cuda hello xy
