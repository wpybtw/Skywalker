all: main main_high_degree
# util sampler main

CUFLAG= -I../include -rdc=true -gencode=arch=compute_75,code=sm_75 -std=c++11 -res-usage  -lineinfo -Xptxas -O0,-v  #-keep   #-Xptxas -O3,-v
debug: CUFLAG +=  -G -g 
debug: main main_high_degree

main: main.cu util.cu sampler.cu
	nvcc main.cu util.cu sampler.cu  $(CUFLAG) -o main

main_high_degree: main_high_degree.cu util.cu sampler_high_degree.cu
	nvcc main_high_degree.cu util.cu sampler_high_degree.cu  $(CUFLAG) -o main_high_degree

# main: main.cu util.o sampler.o
# 	nvcc main.cu util.o sampler.o  $(CUFLAG) -o main



util: ../include/util.cuh util.cu
	nvcc util.cu  $(CUFLAG) -c 

sampler: ../include/sampler.cuh sampler.cu util.cu
	nvcc sampler.cu util.o  $(CUFLAG) -c 

sampler_high_degree: ../include/sampler.cuh sampler_high_degree.cu util.cu
	nvcc sampler_high_degree.cu util.o  $(CUFLAG) -c 


test: main
	#Multidimensional random walk
	./main wg ../others/C-SAW/non-stream/WG/beg.bin ../others/C-SAW/non-stream/WG/csr.bin 315 32 4000 2000 1 2000 1




lj: main
	./main lj ~/data/soc-LiveJournal1.txt_beg_pos.bin  ~/data/soc-LiveJournal1.txt_csr.bin   100 32 4000 2 1 1 1

# alias_table: ../include/alias_table.cuh alias_table.cu
# 	nvcc alias_table.cu  $(CUFLAG)   -c 

# alias: tmp.cu util.o alias_table.o #-arch=sm_75
# 	nvcc tmp.cu util.o alias_table.o $(CUFLAG)  -o alias


clean:
	rm *.o  main main_high_degree alias *.cubin *.fatbin *.fatbin.*  *.reg.cu *.ii *.gpu *.stub.c *.module_id *.cudafe1.* *.reg.c *.ptx