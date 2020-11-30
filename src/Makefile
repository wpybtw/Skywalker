all: main_gbuffer   sample sample_rw  sample_rw2 # rw main_spliced
# util sampler main

CUFLAG= -I../include -rdc=true -gencode=arch=compute_75,code=sm_75 -std=c++11 -res-usage  -lineinfo  -Xptxas -v  #-keep   #-Xptxas -O3,-v   
debug: CUFLAG +=  -G -g 
debug: main_gbuffer sample	sample_rw  sample_rw2

# main: main.cu util.cu sampler.cu
# 	nvcc main.cu util.cu sampler.cu  $(CUFLAG) -o main

main_spliced: main.cu util.cu sampler_spliced.cu kernel.cu
	nvcc main.cu util.cu sampler_spliced.cu kernel.cu $(CUFLAG) -o main_spliced

main_gbuffer: main.cu util.cu sampler_gbuffer.cu kernel.cu
	nvcc main.cu util.cu sampler_gbuffer.cu kernel.cu $(CUFLAG) -o main_gbuffer

sample: entry.cu util.cu createTable.cu sample.cu kernel.cu vec.cu
	nvcc entry.cu util.cu createTable.cu sample.cu kernel.cu vec.cu $(CUFLAG) -o sample

sample_rw: entry.cu util.cu createTable.cu sample_rw.cu kernel.cu vec.cu
	nvcc entry.cu util.cu createTable.cu sample_rw.cu kernel.cu vec.cu $(CUFLAG) -o sample_rw

sample_rw2: entry_rw.cu util.cu createTable.cu sample_rw2.cu kernel.cu vec.cu
	nvcc entry_rw.cu util.cu createTable.cu sample_rw2.cu kernel.cu vec.cu $(CUFLAG) -o sample_rw2

# rw: walker.cu util.cu rw.cu kernel.cu
# 	nvcc walker.cu util.cu kernel.cu rw.cu  $(CUFLAG) -o rw

# main: main.cu util.o sampler.o
# 	nvcc main.cu util.o sampler.o  $(CUFLAG) -o main



# util: ../include/util.cuh util.cu
# 	nvcc util.cu  $(CUFLAG) -c 

# sampler: ../include/sampler.cuh sampler.cu util.cu
# 	nvcc sampler.cu util.o  $(CUFLAG) -c 

# sampler_high_degree: ../include/sampler.cuh sampler_high_degree.cu util.cu
# 	nvcc sampler_high_degree.cu util.o  $(CUFLAG) -c 


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
	rm *.o  main main_spliced main_gbuffer createTable sample_rw sample_rw2 alias *.cubin *.fatbin *.fatbin.*  *.reg.cu *.ii *.gpu *.stub.c *.module_id *.cudafe1.* *.reg.c *.ptx