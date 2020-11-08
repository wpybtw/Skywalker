all: main

CUFLAG= -I../include -G -g -rdc=true -gencode=arch=compute_75,code=sm_75 -std=c++11

main: main.cu
	nvcc main.cu  $(CUFLAG) -o main

# alias_table: alias_table.cuh alias_table.cu
# 	nvcc alias_table.cu  $(CUFLAG)   -c 


# alias: tmp.cu util.o alias_table.o #-arch=sm_75
# 	nvcc tmp.cu util.o alias_table.o $(CUFLAG)  -o alias

test:main
	#Multidimensional random walk
	./main wg ../others/C-SAW/non-stream/WG/beg.bin ../others/C-SAW/non-stream/WG/csr.bin 315 32 4000 2000 1 2000 1

clean:
	rm *.o