#!/bin/bash
./build/skywalker --ngpu=1 --s=1 --static=0 --bias=1 --ol=0 --d=100 --n=2000 --dw=1 --rw=1 --k=1 --m=4 --printresult=1 --umresult=1 --input ./wiki-Vote.gr -v