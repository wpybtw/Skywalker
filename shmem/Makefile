all: util alias_table alias

CUFLAG= -G -g -rdc=true -gencode=arch=compute_75,code=sm_75

util: util.cuh util.cu
	nvcc util.cu  $(CUFLAG) -c 

alias_table: alias_table.cuh alias_table.cu
	nvcc alias_table.cu  $(CUFLAG)   -c 


alias: tmp.cu util.o alias_table.o #-arch=sm_75
	nvcc tmp.cu util.o alias_table.o $(CUFLAG)  -o alias

clean:
	rm *.o