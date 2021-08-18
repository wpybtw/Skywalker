#!/bin/bash -x
###
# @Description:
# @Date: 2020-11-17 13:39:45
# @LastEditors: Pengyu Wang
# @LastEditTime: 2021-01-15 15:49:20
###

DATA=(lj)
HD=(0.5)
NV=(4847571)

# DATA=(web-Google lj orkut arabic-2005 uk-2005 sk-2005 friendster) # uk-union rmat29 web-ClueWeb09) eu-2015-host-nat twitter-2010
# HD=(0.25 0.5 1 0.25 0.25 0.5 1)                                   # uk-union rmat29 web-ClueWeb09)
# NV=(916428 4847571 3072627 39459923 22744077 50636151 124836180)

# DATA=( sk-2005 friendster)
# HD=(   4  1 )
ITR=1
NG=4 #8

GR=".w.gr"
EXE="./bin/main" #main_degree
SG="--ngpu=1 --s"
RW="--deepwalk "
SP="--sage "
BATCH="--n 40000 -v"

ROOT_DIR=$PWD
LOG_FILE=${ROOT_DIR}"/result/fig8_biased.csv"

# DATA_DIR="~/data"
DATA_DIR=${ROOT_DIR}"/dataset"
GraphWalker_DIR="/home/pywang/sampling/GraphWalker"
KnightKing_DIR="/home/pywang/sampling/KnightKing"
CSAW_DIR="/home/pywang/sampling/C-SAW"

# --randomweight=1 --weightrange=2

echo "-------------------------------------------------------Skywalker offline rw 100  ${BATCH}" >>"${LOG_FILE}"
for idx in $(seq 1 ${#DATA[*]}); do
    for i in $(seq 1 ${ITR}); do
        ./bin/main -bias=1 --ol=0 ${SG} ${RW} --input ~/data/${DATA[idx - 1]}${GR} --hd=${HD[idx - 1]} ${BATCH} >>"${LOG_FILE}"
    done
done

echo "-------------------------------------------------------Skywalker offline ppr 0.15  ${BATCH}" >>"${LOG_FILE}"
for idx in $(seq 1 ${#DATA[*]}); do
    for i in $(seq 1 ${ITR}); do
        ./bin/main -bias=1 --ol=0 ${RW} --tp=0.15 --input ~/data/${DATA[idx - 1]}${GR} --hd=${HD[idx - 1]} ${BATCH} ${SG} >>"${LOG_FILE}"
    done
done

echo "-------------------------------------------------------Skywalker node2vec ${POLICY} " >>"${LOG_FILE}"
for idx in $(seq 1 ${#DATA[*]}); do
    for i in $(seq 1 ${ITR}); do
        ./bin/node2vec -node2vec ${RW} --input ~/data/${DATA[idx - 1]}${GR} --hd=${HD[idx - 1]} ${BATCH} ${SG} ${POLICY} >>"${LOG_FILE}"
    done
done

echo "-------------------------------------------------------Skywalker offline sp sage ${BATCH}" >>"${LOG_FILE}"
for idx in $(seq 1 ${#DATA[*]}); do
    for i in $(seq 1 ${ITR}); do
        ./bin/main -bias=1 --ol=0 ${SG} --rw=0 --sage --input ~/data/${DATA[idx - 1]}${GR} --hd=${HD[idx - 1]} ${BATCH} >>"${LOG_FILE}"
    done
done

echo "-------------------Runtime of C-SAW  need to be scale by 10 due to 4k as batch size. And scale by sampled edges ratio-------------------" >>"${LOG_FILE}"

echo "----------------------C-SAW  biased walk 4k 64-------------------" >>"${LOG_FILE}"
for idx in $(seq 1 ${#DATA[*]}); do
    echo "------------"${DATA[idx - 1]}
    $CSAW_DIR/non-stream/sampling.bin wg ~/data/${DATA[idx - 1]}.w.edge_beg_pos.bin ~/data/${DATA[idx - 1]}.w.edge_csr.bin 100 32 4000 1 1 100 1 >>"${LOG_FILE}"
done

echo "----------------------C-SAW  sampling biased 4k 20 2 64-------------------" >>"${LOG_FILE}"
for idx in $(seq 1 ${#DATA[*]}); do
    echo "------------"${DATA[idx - 1]}
    $CSAW_DIR/non-stream/sampling.bin wg ~/data/${DATA[idx - 1]}.w.edge_beg_pos.bin ~/data/${DATA[idx - 1]}.w.edge_csr.bin 100 32 4000 1 20 2 1 >>"${LOG_FILE}"
done

echo "----------------------KnightKing biased_walk  -------------------" >>"${LOG_FILE}"
for idx in $(seq 1 ${#DATA[*]}); do
    echo "------------"${DATA[idx - 1]}
    $KnightKing_DIR/build/bin/biased_walk -w 40000 -g ~/data/${DATA[idx - 1]}.data -v ${NV[idx - 1]} -l 100 >>"${LOG_FILE}"
done

echo "----------------------KnightKing biased node2vec-------------------" >>"${LOG_FILE}"
for idx in $(seq 1 ${#DATA[*]}); do
    echo "------------"${DATA[idx - 1]}
    $KnightKing_DIR/build/bin/node2vec -w 40000 -l 100 -s weighted -p 2.0 -q 0.5 -g ~/data/${DATA[idx - 1]}.data -v ${NV[idx - 1]} >>"${LOG_FILE}"
done

echo "----------------------KnightKing ppr biased -------------------" >>"${LOG_FILE}"
for idx in $(seq 1 ${#DATA[*]}); do
    echo "------------"${DATA[idx - 1]}
    $KnightKing_DIR/build/bin/ppr -s weighted -t 0.15 -w 40000 -g ~/data/${DATA[idx - 1]}.data -v ${NV[idx - 1]} >>"${LOG_FILE}"
done
