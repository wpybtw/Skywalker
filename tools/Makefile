all: getDegree
# util sampler main

CUFLAG= -I../include -rdc=true -gencode=arch=compute_75,code=sm_75 -std=c++11 -res-usage  -lineinfo  -Xptxas -v  #-keep   #-Xptxas -O3,-v   
debug: CUFLAG +=  -G -g 
debug: main main_gbuffer main_spliced	

# main: main.cu util.cu sampler.cu

getDegree: getDegree.cu ../src/util.cu 
	nvcc getDegree.cu ../src/util.cu   $(CUFLAG) -o getDegree

