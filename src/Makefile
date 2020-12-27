all: main main_degree node2vec
# util sampler main


debug: CUFLAG +=  -G -g 
debug: LDFLAGS +=  -G -g 
debug: main  main_degree node2vec
SRC_DIR:= .
OBJ_DIR:= obj
BIN_DIR:= bin

SRC_FILES := $(wildcard $(SRC_DIR)/*.cu)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SRC_FILES))

# SRC_FILES += $(wildcard $(SRC_DIR)/api/*.cu)
API_SRC_DIR:= api
API_OBJ_DIR:= obj/api
API_SRC_FILES := $(wildcard $(API_SRC_DIR)/*.cu)
API_OBJ_FILES := $(patsubst $(API_SRC_DIR)/%.cu,$(API_OBJ_DIR)/%.o,$(API_SRC_FILES))

LDFLAGS := -Xcompiler -fopenmp -Xlinker -lgomp /home/pywang/sampling/build/deps/gflags/libgflags_nothreads.a  -Ldeps/gflags -gencode=arch=compute_75,code=sm_75
CUFLAG= -I../include -I/home/pywang/sampling/build/deps/gflags/include -rdc=true -gencode=arch=compute_75,code=sm_75 -std=c++11  -lineinfo  -Xptxas -v -Xcompiler -fopenmp  #-keep   #-Xptxas -O3,-v   

main_degree: $(OBJ_FILES)  $(API_OBJ_DIR)/bias_degree.o
	nvcc $(LDFLAGS) -o $(BIN_DIR)/$@ $^

main: $(OBJ_FILES)  $(API_OBJ_DIR)/bias_static.o
	nvcc $(LDFLAGS) -o $(BIN_DIR)/$@ $^

node2vec: $(OBJ_FILES) $(API_OBJ_DIR)/bias_node2vec.o
	nvcc $(LDFLAGS) -o $(BIN_DIR)/$@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu 
	nvcc  $(CUFLAG) -c -o $@ $<

$(API_OBJ_DIR)/%.o: $(API_SRC_DIR)/%.cu 
	@mkdir -p $(API_OBJ_DIR)
	nvcc  $(CUFLAG) -c -o $@ $<
	
test: main
	#Multidimensional random walk
	./main wg ../others/C-SAW/non-stream/WG/beg.bin ../others/C-SAW/non-stream/WG/csr.bin 315 32 4000 2000 1 2000 1

lj: main
	./main lj ~/data/soc-LiveJournal1.txt_beg_pos.bin  ~/data/soc-LiveJournal1.txt_csr.bin   100 32 4000 2 1 1 1

clean:
	rm $(OBJ_DIR)/*.o  $(API_OBJ_DIR)/*.o  $(BIN_DIR)/*
	# main  main_degree node2vec
	# rm *.cubin *.fatbin *.fatbin.*  *.reg.cu *.ii *.gpu *.stub.c *.module_id *.cudafe1.* *.reg.c *.ptx