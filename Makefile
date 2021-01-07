all: main main_degree node2vec

debug: CUFLAG +=  -G -g -DDEBUG
debug: LDFLAGS +=  -G -g -DDEBUG
debug: main  main_degree node2vec

SRC_DIR:= src
OBJ_DIR:= bin/obj
BIN_DIR:= bin

SRC_FILES := $(wildcard $(SRC_DIR)/*.cu)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SRC_FILES))

API_SRC_DIR:= src/api
API_OBJ_DIR:= bin/obj/api
API_SRC_FILES := $(wildcard $(API_SRC_DIR)/*.cu)
API_OBJ_FILES := $(patsubst $(API_SRC_DIR)/%.cu,$(API_OBJ_DIR)/%.o,$(API_SRC_FILES))

FLAGS= -Xcompiler -fopenmp -lineinfo -gencode=arch=compute_75,code=sm_75 -DNDEBUG# -Xptxas -v  -Xcompiler -Werror

# 
LDFLAGS := ${FLAGS} -Xlinker -lgomp -Xlinker -lnuma  ./build/deps/gflags/libgflags_nothreads.a  -Ldeps/gflags 
CUFLAG= ${FLAGS} -I./include -I./build/deps/gflags/include -rdc=true -std=c++11  #-keep   #-Xptxas -O3,-v   

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
	./bin/main --k 2 --d 2 --ol=1  --input ~/data/lj.w.gr --ngpu=4 --hd=1 --n=400000

# lj: main
# 	./main lj ~/data/soc-LiveJournal1.txt_beg_pos.bin  ~/data/soc-LiveJournal1.txt_csr.bin   100 32 4000 2 1 1 1

clean:
	rm $(OBJ_DIR)/*.o  $(API_OBJ_DIR)/*.o  $(BIN_DIR)/*
	# main  main_degree node2vec
	# rm *.cubin *.fatbin *.fatbin.*  *.reg.cu *.ii *.gpu *.stub.c *.module_id *.cudafe1.* *.reg.c *.ptx