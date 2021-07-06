#!/bin/bash -x
###
 # @Description: 
 # @Date: 2020-11-17 13:39:45
 # @LastEditors: Pengyu Wang
 # @LastEditTime: 2021-01-15 16:43:38
 # @FilePath: /skywalker/figs/online.sh
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
BATCH="--n 40000 "
POLICY="--static=0"
OUTPUT=" online.csv "

# --randomweight=1 --weightrange=2 

# echo "-------------------------------------------------------online rw 100 ${POLICY} " >> ${OUTPUT}
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     for i in $(seq 1  ${ITR})
#     do
#         ./bin/main -bias=1 --ol=1 ${SG} ${RW} --input ~/data/${DATA[idx-1]}${GR} --hd=${HD[idx-1]} ${BATCH} ${POLICY} >>  ${OUTPUT}
#     done
# done

# echo "-------------------------------------------------------online ppr 0.15" >> online.csv
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     for i in $(seq 1  ${ITR})
#     do
#         ./bin/main  -bias=1 --ol=1  ${RW}  --tp=0.15   --input ~/data/${DATA[idx-1]}${GR} --hd=${HD[idx-1]} ${BATCH} ${SG} >> online.csv
#     done
# done

echo "-------------------------------------------------------node2vec ${POLICY} " >>  ${OUTPUT}
for idx in $(seq 1 ${#DATA[*]}) 
do
    for i in $(seq 1  ${ITR})
    do
        ./bin/node2vec  -node2vec  ${RW}  --input ~/data/${DATA[idx-1]}${GR} --hd=${HD[idx-1]} ${BATCH} ${SG} ${POLICY} >> ${OUTPUT}
    done
done

# echo "-------------------------------------------------------online sp 100 ${POLICY} " >> ${OUTPUT}
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     for i in $(seq 1  ${ITR})
#     do
#         ./bin/main -bias=1 --ol=1 ${SG} ${SP} --input ~/data/${DATA[idx-1]}${GR} --hd=${HD[idx-1]} ${BATCH} ${POLICY} >>  ${OUTPUT}
#     done
# done






