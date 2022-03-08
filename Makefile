all: build
debug: debug


.PHONY:build debug test

SRC_DIR:= src
SRC_FILES := $(wildcard $(SRC_DIR)/*.cu)
HEADER_FILES := $(wildcard include/*.cuh)

build: 
	-mkdir build;cd build;cmake ..;make -j

debug: 
	-mkdir build;cd build;cmake .. -DCMAKE_BUILD_TYPE=Debug;make -j
	
test: 
	./build/skywalker --k 2 --d 2 --ol=1  --input ~/data/lj.w.gr --ngpu=4 --hd=1 --n=400000
	./build/skywalker  -bias=1 --ol=1 --ngpu=1 --s --sage --input ~/data/orkut.w.gr  -v
	./build/skywalker  -bias=1 --ol=0 --ngpu=1 --s --sage --input ~/data/orkut.w.gr  -v


clean:
	cd build;make clean