#!/bin/bash -x
###
 # @Description: 
 # @Date: 2020-11-17 13:39:45
 # @LastEditors: Please set LastEditors
 # @LastEditTime: 2021-01-15 16:45:34
 # @FilePath: /skywalker/figs/scale.sh
### 
DATA=(web-Google lj orkut arabic-2005 uk-2005  sk-2005 friendster) # uk-union rmat29 web-ClueWeb09) eu-2015-host-nat twitter-2010
HD=(0.25          0.5  1     0.25        0.25      0.5           1) # uk-union rmat29 web-ClueWeb09)
NV=(916428    4847571 3072627  39459923   22744077     50636151 124836180)
# HD=(4             2   1     4         4       2           1) # uk-union rmat29 web-ClueWeb09)

# DATA=( sk-2005 friendster) 
# HD=(   4  1 )
ITR=1
NG=8 #8


GR=".w.gr"
EXE="./bin/main" #main_degree
SG="--ngpu=1 --s"
RW="--rw=1 --k 1 --d 100 "
SP="--rw=0 --k 20 --d 2 "
BATCH="--n 40000"

# --randomweight=1 --weightrange=2 

echo "-------------------------------------------------------offline rw 100" >> scale.csv
for idx in $(seq 1 ${#DATA[*]}) 
do
    for i in $(seq 1  ${NG})
    do
        ./bin/main -bias=1 --ol=0 --ngpu=$i --s ${RW} --input ~/data/${DATA[idx-1]}${GR} --hd=${HD[idx-1]} ${BATCH} >> scale.csv
    done
done

echo "-------------------------------------------------------offline ppr 0.15" >> scale.csv
for idx in $(seq 1 ${#DATA[*]}) 
do
    for i in $(seq 1  ${NG})
    do
        ./bin/main  -bias=1 --ol=0 --n=40000 ${RW}  --tp=0.15   --input ~/data/${DATA[idx-1]}${GR} --hd=${HD[idx-1]} ${BATCH} --ngpu=$i --s >> scale.csv
    done
done


echo "-------------------------------------------------------offline sp 100" >> scale.csv
for idx in $(seq 1 ${#DATA[*]}) 
do
    for i in $(seq 1  ${NG})
    do
        ./bin/main -bias=1 --ol=0 --ngpu=$i --s ${SP} --input ~/data/${DATA[idx-1]}${GR} --hd=${HD[idx-1]} ${BATCH} >> scale.csv
    done
done






echo "-------------------------------------------------------online rw 100" >> scale.csv
for idx in $(seq 1 ${#DATA[*]}) 
do
    for i in $(seq 1  ${NG})
    do
        ./bin/main -bias=1 --ol=1 --ngpu=$i --s ${RW} --input ~/data/${DATA[idx-1]}${GR} --hd=${HD[idx-1]} ${BATCH} >> scale.csv
    done
done

echo "-------------------------------------------------------online ppr 0.15" >> scale.csv
for idx in $(seq 1 ${#DATA[*]}) 
do
    for i in $(seq 1  ${NG})
    do
        ./bin/main  -bias=1 --ol=1 --n=40000 ${RW}  --tp=0.15   --input ~/data/${DATA[idx-1]}${GR} --hd=${HD[idx-1]} ${BATCH} --ngpu=$i --s >> scale.csv
    done
done

echo "-------------------------------------------------------online ppr 0.15" >> scale.csv
for idx in $(seq 1 ${#DATA[*]}) 
do
    for i in $(seq 1  ${NG})
    do
        ./bin/node2vec  -node2vec --n=40000 ${RW}  --input ~/data/${DATA[idx-1]}${GR} --hd=${HD[idx-1]} ${BATCH} --ngpu=$i --s >> scale.csv
    done
done

echo "-------------------------------------------------------online sp 100" >> scale.csv
for idx in $(seq 1 ${#DATA[*]}) 
do
    for i in $(seq 1  ${NG})
    do
        ./bin/main -bias=1 --ol=1 --ngpu=$i --s ${SP} --input ~/data/${DATA[idx-1]}${GR} --hd=${HD[idx-1]} ${BATCH} >> scale.csv
    done
done

