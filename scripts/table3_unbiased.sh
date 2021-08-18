#!/bin/bash  -x
###
# @Description:
# @Date: 2020-11-17 13:39:45
# @LastEditors: Pengyu Wang
# @LastEditTime: 2021-01-17 21:38:38
# @FilePath: /skywalker/figs/unbiased.sh
###

# using sample dataset
DATA=(lj)
HD=(0.5)
NV=(4847571)


# DATA=(web-Google lj orkut arabic-2005 uk-2005 sk-2005 friendster) # uk-union rmat29 web-ClueWeb09) eu-2015-host-nat twitter-2010
# HD=(0.25 0.5 1 0.25 0.25 0.5 1)                                   # uk-union rmat29 web-ClueWeb09)
# NV=(916428 4847571 3072627 39459923 22744077 50636151 124836180)
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
BATCH="--n=40000 -v"

ROOT_DIR=$PWD
LOG_FILE=${ROOT_DIR}"/result/table3_unbiased.csv"

# DATA_DIR="~/data"
DATA_DIR=${ROOT_DIR}"/dataset"
GraphWalker_DIR="/home/pywang/sampling/GraphWalker"
KnightKing_DIR="/home/pywang/sampling/KnightKing"
CSAW_DIR="/home/pywang/sampling/C-SAW"
NEXTDOOR_DIR="/home/pywang/sampling/nextdoor-experiments"

echo "-------------------------------------------------------Skywalker unbias rw 100" >>"${LOG_FILE}"
for idx in $(seq 1 ${#DATA[*]}); do
    ./bin/main --bias=0 --input $DATA_DIR/${DATA[idx - 1]}${GR} --ngpu 1 ${RW} ${BATCH} >>"${LOG_FILE}"
done

echo "-------------------------------------------------------Skywalker unbias ppr 100" >>"${LOG_FILE}"
for idx in $(seq 1 ${#DATA[*]}); do
    ./bin/main --bias=0 --input $DATA_DIR/${DATA[idx - 1]}${GR} --ngpu 1 --tp=0.15 ${RW} ${BATCH} >>"${LOG_FILE}"
done

echo "-------------------------------------------------------Skywalker unbias node2vec" >>"${LOG_FILE}"
for idx in $(seq 1 ${#DATA[*]}); do
    ./bin/main --bias=0 --ol=0 --buffer --input $DATA_DIR/${DATA[idx - 1]}${GR} --ngpu 1 --node2vec ${BATCH} >>"${LOG_FILE}"
done

echo "-------------------------------------------------------Skywalker unbias sage 40k" >>"${LOG_FILE}"
for idx in $(seq 1 ${#DATA[*]}); do
    ./bin/main --bias=0 --input $DATA_DIR/${DATA[idx - 1]}${GR} --ngpu 1 --sage ${BATCH} >>"${LOG_FILE}"
done

echo "----------------------KnightKing unbiased 40k degree-------------------" >>"${LOG_FILE}"
for idx in $(seq 1 ${#DATA[*]}); do
    echo ${DATA[idx - 1]} >>"${LOG_FILE}"
    $KnightKing_DIR/build/bin/deepwalk -w 40000 -l 100 -s unweighted -g $DATA_DIR/${DATA[idx - 1]}.uw.data -v ${NV[idx - 1]} >>"${LOG_FILE}"
done
echo "----------------------KnightKing unbiased node2vec-------------------" >>"${LOG_FILE}"
for idx in $(seq 1 ${#DATA[*]}); do
    echo ${DATA[idx - 1]} >>"${LOG_FILE}"
    $KnightKing_DIR/build/bin/node2vec -w 40000 -l 100 -s unweighted -p 2.0 -q 0.5 -g $DATA_DIR/${DATA[idx - 1]}.uw.data -v ${NV[idx - 1]} >>"${LOG_FILE}"
done

echo "----------------------KnightKing ppr unbiased ------------------" >>"${LOG_FILE}"
for idx in $(seq 1 ${#DATA[*]}); do
    echo ${DATA[idx - 1]} >>"${LOG_FILE}"
    $KnightKing_DIR/build/bin/ppr -w 40000 -s unweighted -t 0.15 -v ${NV[idx - 1]} -g $DATA_DIR/${DATA[idx - 1]}.uw.data >>"${LOG_FILE}"
done

echo "----------------------nextdoor node2vec -------------------" >>"${LOG_FILE}"
for idx in $(seq 1 ${#DATA[*]}); do
    echo "------------"${DATA[idx - 1]} >>"${LOG_FILE}"
    $NEXTDOOR_DIR/NextDoor/src/apps/randomwalks/Node2VecSampling -g $DATA_DIR/${DATA[idx - 1]}.data -t edge-list -f binary -n 1 -k TransitParallel -l >>"${LOG_FILE}"
done
echo "----------------------nextdoor kh sample-------------------" >>"${LOG_FILE}"
for idx in $(seq 1 ${#DATA[*]}); do
    echo "------------"${DATA[idx - 1]} >>"${LOG_FILE}"
    $NEXTDOOR_DIR/NextDoor/src/apps/khop/KHopSampling -g $DATA_DIR/${DATA[idx - 1]}.data -t edge-list -f binary -n 1 -k TransitParallel -l >>"${LOG_FILE}"
done

ED=".w.edge"
EXE="./bin/apps/rwdomination"                    #main_degree

cd $GraphWalker_DIR
echo "------------skiping web-Google and orkut for GraphWalker due to internal errors"
echo "-------------------------------------------------------GraphWalker unbias rw 40000 100" >>"${LOG_FILE}"
for idx in $(seq 1 ${#DATA[*]}); do
    if [ ${DATA[idx - 1]} != "web-Google" -a ${DATA[idx - 1]} != "orkut" ]    
    then  
    ./bin/apps/rawrandomwalks file  ~/data/${DATA[idx - 1]}.w.edge R 40000 L 100 N ${NV[idx - 1]} >>"${LOG_FILE}"  
    fi
done

echo "-------------------------------------------------------GraphWalker unbias ppr 40000 100" >>"${LOG_FILE}"
for idx in $(seq 1 ${#DATA[*]}); do
    if [ ${DATA[idx - 1]} != "web-Google" -a ${DATA[idx - 1]} != "orkut" ]  
    then
    ./bin/apps/msppr file ~/data/${DATA[idx - 1]}.w.edge firstsource 0 numsources 40000 walkspersource 1 maxwalklength 100 prob 0.15 >>"${LOG_FILE}"
    fi
done
