#!/bin/bash -x
###
 # @Description: 
 # @Date: 2020-11-17 13:39:45
 # @LastEditors: Please set LastEditors
 # @LastEditTime: 2021-01-15 15:49:20
 # @FilePath: /skywalker/figs/offline.sh
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
RW="--deepwalk "
SP="--sage "
BATCH="--n 40000 "
LOG_FILE="offline.csv"





# --randomweight=1 --weightrange=2 

# echo "-------------------------------------------------------offline rw 100  ${BATCH}" >> ${LOG_FILE}
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     for i in $(seq 1  ${ITR})
#     do
#         ./bin/main -bias=1 --ol=0 ${SG} ${RW} --input ~/data/${DATA[idx-1]}${GR} --hd=${HD[idx-1]} ${BATCH} >> ${LOG_FILE} 
#     done
# done

# echo "-------------------------------------------------------offline ppr 0.15  ${BATCH}" >> ${LOG_FILE}
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     for i in $(seq 1  ${ITR})
#     do
#         ./bin/main  -bias=1 --ol=0  ${RW}  --tp=0.15   --input ~/data/${DATA[idx-1]}${GR} --hd=${HD[idx-1]} ${BATCH} ${SG} >> ${LOG_FILE}
#     done
# done


# echo "-------------------------------------------------------offline sp 100${BATCH}" >> ${LOG_FILE}
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     for i in $(seq 1  ${ITR})
#     do
#         ./bin/main -bias=1 --ol=0 ${SG} ${SP} --input ~/data/${DATA[idx-1]}${GR} --hd=${HD[idx-1]} ${BATCH} >> ${LOG_FILE}
#     done
# done

echo "-------------------------------------------------------offline sp 20 20 ${BATCH}" >> ${LOG_FILE}
for idx in $(seq 1 ${#DATA[*]}) 
do
    for i in $(seq 1  ${ITR})
    do
        ./bin/main -bias=1 --ol=0 ${SG} --rw=0 --k=20 --d=2 --input ~/data/${DATA[idx-1]}${GR} --hd=${HD[idx-1]} ${BATCH} >> ${LOG_FILE}
    done
done





