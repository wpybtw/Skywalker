#!/bin/bash  -x
###
 # @Description: 
 # @Date: 2020-11-17 13:39:45
 # @LastEditors: Pengyu Wang
 # @LastEditTime: 2021-01-17 21:38:38
 # @FilePath: /skywalker/figs/unbiased.sh
### 
DATA=(web-Google lj orkut arabic-2005 uk-2005  sk-2005 friendster) # uk-union rmat29 web-ClueWeb09) eu-2015-host-nat twitter-2010
HD=(0.25          0.5  1     0.25        0.25      0.5           1) # uk-union rmat29 web-ClueWeb09)
NV=(916428    4847571 3072627  39459923   22744077     50636151 124836180)
# HD=(4             2   1     4         4       2           1) # uk-union rmat29 web-ClueWeb09)

# DATA=( sk-2005 friendster) 
# HD=(   4  1 )
ITR=1
NG=4 #8

GR=".w.gr"
EXE="./bin/main" #main_degree
SG="--ngpu=1 --s"
RW="--rw=1 --k 1 --d 100 "
SP="--rw=0 --k 20 --d 2 "
BATCH="--n=40000"
OUT='>> ./figs/result/dynamic.csv'

# --randomweight=1 --weightrange=2 


# walker
echo "-------------------------------------------------------unbias rw 100" >> ./figs/result/dynamic.csv
for idx in $(seq 1 ${#DATA[*]}) 
do
    ./bin/main --bias=0  --input ~/data/${DATA[idx-1]}${GR}  --ngpu 1 ${RW} ${BATCH} >> ./figs/result/dynamic.csv
done

echo "-------------------------------------------------------unbias rw 100 dynamic" >> ./figs/result/dynamic.csv
for idx in $(seq 1 ${#DATA[*]}) 
do
    ./bin/main --bias=0  --input ~/data/${DATA[idx-1]}${GR}  --ngpu 1 ${RW} ${BATCH} --dynamic=1>> ./figs/result/dynamic.csv
done

# echo "-------------------------------------------------------unbias sp" >> ./figs/result/dynamic.csv
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     ./bin/main --bias=0  --input ~/data/${DATA[idx-1]}${GR}  --ngpu 1 ${SP} ${BATCH} >> ./figs/result/dynamic.csv
# done

