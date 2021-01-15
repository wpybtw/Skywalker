###
 # @Author: Pengyu Wang
 # @Date: 2021-01-15 14:35:15
 # @LastEditTime: 2021-01-15 14:38:16
 # @Description: 
 # @FilePath: /skywalker/scripts/mem_test.sh
### 
DATA=(web-Google lj orkut arabic-2005 uk-2005  sk-2005 friendster) # uk-union rmat29 web-ClueWeb09) eu-2015-host-nat twitter-2010
HD=(0.25          0.5  1     0.25        0.25      0.5           1) # uk-union rmat29 web-ClueWeb09)
NV=(916428    4847571 3072627  39459923   22744077     50636151 124836180)
# HD=(4             2   1     4         4       2           1) # uk-union rmat29 web-ClueWeb09)

# DATA=( sk-2005 friendster) 
# HD=(   4  1 )
ITR=1
NG=4

GR=".w.gr"
EXE="./bin/main" #main_degree
SG="--ngpu=1 --s"

# node2vec always online
# export OMP_PROC_BIND=TRUE
# GOMP_CPU_AFFINITY="0-9 10-19 20-29 30-99"
# OMP_PLACES=cores
# OMP_PROC_BIND=close
# correct one
# OMP_PLACES=cores OMP_PROC_BIND=spread
# --randomweight=1 --weightrange=2 



echo "-------------------------------------------------------unbias sample 2 20 40k"
for idx in $(seq 1 ${#DATA[*]}) 
do
    ./bin/main  --input ~/data/${DATA[idx-1]}${GR} --d 2 --k 20 --n 40000 --bias=0 --rw=0 --ngpu=1  --ol=0 --umgraph=1 -v
    ./bin/main  --input ~/data/${DATA[idx-1]}${GR} --d 2 --k 20 --n 40000 --bias=0 --rw=0 --ngpu=1  --ol=0 --hmgraph=1 -v
    ./bin/main  --input ~/data/${DATA[idx-1]}${GR} --d 2 --k 20 --n 40000 --bias=0 --rw=0 --ngpu=1  --ol=0 --gmgraph=1 --gmid=1 -v
done


