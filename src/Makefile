all: main   
# util sampler main


debug: CUFLAG +=  -G -g 
debug: main 
SRC_DIR:= .
OBJ_DIR:= .
SRC_FILES := $(wildcard $(SRC_DIR)/*.cu)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SRC_FILES))
LDFLAGS := /home/pywang/sampling/build/deps/gflags/libgflags_nothreads.a  -Ldeps/gflags
CUFLAG= -I../include -I/home/pywang/sampling/build/deps/gflags/include -rdc=true -gencode=arch=compute_75,code=sm_75 -std=c++11 -res-usage  -lineinfo  -Xptxas -v    #-keep   #-Xptxas -O3,-v   

main: $(OBJ_FILES)
	nvcc $(LDFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	nvcc  $(CUFLAG) -c -o $@ $<

# main_gbuffer: main.cu util.cu sampler_gbuffer.cu kernel.cu createTable.cu sample_rw.cu sample_rw2.cu
# 	nvcc main.cu util.cu sampler_gbuffer.cu kernel.cu createTable.cu sample_rw.cu sample_rw2.cu $(CUFLAG) -o main_gbuffer

# sample: entry.cu util.cu createTable.cu sample.cu kernel.cu vec.cu
# 	nvcc entry.cu util.cu createTable.cu sample.cu kernel.cu vec.cu $(CUFLAG) -o sample

# sample_rw: entry.cu util.cu createTable.cu sample_rw.cu kernel.cu vec.cu
# 	nvcc entry.cu util.cu createTable.cu sample_rw.cu kernel.cu vec.cu $(CUFLAG) -o sample_rw

# sample_rw2: entry_rw.cu util.cu createTable.cu sample_rw2.cu kernel.cu vec.cu
# 	nvcc entry_rw.cu util.cu createTable.cu sample_rw2.cu kernel.cu vec.cu $(CUFLAG) -o sample_rw2


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
	rm *.o  main *.cubin *.fatbin *.fatbin.*  *.reg.cu *.ii *.gpu *.stub.c *.module_id *.cudafe1.* *.reg.c *.ptx