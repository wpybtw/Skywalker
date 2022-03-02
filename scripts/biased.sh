DATA=(web-Google lj orkut arabic-2005 uk-2005 sk-2005 friendster) # uk-union rmat29 web-ClueWeb09) eu-2015-host-nat twitter-2010
HD=(0.25 0.5 1 0.25 0.25 0.5 1)                                   # uk-union rmat29 web-ClueWeb09)
NV=(916428 4847571 3072627 39459923 22744077 50636151 124836180)
#HD=(4             2   1     4         4       2           1) # uk-union rmat29 web-ClueWeb09)

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

DATA_DIR="/home/xucheng//data"
#DATA_DIR=${ROOT_DIR}"/dataset"
GraphWalker_DIR="/home/pywang/sampling/GraphWalker"
KnightKing_DIR="/home/pywang/sampling/KnightKing"
CSAW_DIR="/home/pywang/sampling/C-SAW"
NEXTDOOR_DIR="/home/pywang/sampling/nextdoor-experiments"

echo "-------------------------------------------------------Skywalker unbias rw 100" #>>"${LOG_FILE}"
for idx in $(seq 1 ${#DATA[*]}); do
    ./bin/main --bias=1 --input $DATA_DIR/${DATA[idx - 1]}${GR} --ngpu 1 ${RW} ${BATCH} #>>"${LOG_FILE}"
done

echo "-------------------------------------------------------Skywalker unbias ppr 100" #>>"${LOG_FILE}"
for idx in $(seq 1 ${#DATA[*]}); do
    ./bin/main --bias=1 --input $DATA_DIR/${DATA[idx - 1]}${GR} --ngpu 1 --tp=0.15 ${RW} ${BATCH} #>>"${LOG_FILE}"
done

echo "-------------------------------------------------------Skywalker unbias node2vec" #>>"${LOG_FILE}"
for idx in $(seq 1 ${#DATA[*]}); do
    ./bin/main --bias=1 --ol=0 --buffer --input $DATA_DIR/${DATA[idx - 1]}${GR} --ngpu 1 --node2vec ${BATCH} # >>"${LOG_FILE}"
done