###
 # @Description: 
 # @Date: 2020-11-17 13:39:45
 # @LastEditors: PengyuWang
 # @LastEditTime: 2021-01-05 17:45:29
 # @FilePath: /sampling/scripts/test.sh
### 
DATA=(web-Google lj orkut arabic-2005 uk-2005  sk-2005 friendster) # uk-union rmat29 web-ClueWeb09) eu-2015-host-nat twitter-2010
HD=(0.25          0.5  1     0.25        0.25      0.5           1) # uk-union rmat29 web-ClueWeb09)
# HD=(4             2   1     4         4       2           1) # uk-union rmat29 web-ClueWeb09)

# DATA=( sk-2005 friendster) 
# HD=(   4  1 )
ITR=1

GR=".w.gr"
EXE="./bin/main" #main_degree

echo ${EXE}

# export OMP_PROC_BIND=TRUE
# GOMP_CPU_AFFINITY="0-9 10-19 20-29 30-99"
# OMP_PLACES=cores
# OMP_PROC_BIND=close

# correct one
# OMP_PLACES=cores OMP_PROC_BIND=spread

# --randomweight=1 --weightrange=2 


# echo "-------------------------------------------------------unbias rw 4000 100"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     ${EXE} --rw=1 --k 1 --d 100 --ol=1 --bias=0  --input ~/data/${DATA[idx-1]}${GR}
# done

# echo "-------------------------------------------------------online layer sampling 4k 100"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     ${EXE} --rw=0 --k 1 --d 100 --ol=1 --input ~/data/${DATA[idx-1]}${GR} --hd=${HD[idx-1]}
# done


echo "-------------------------------------------------------online walk 400k 100"
for idx in $(seq 1 ${#DATA[*]}) 
do
    for i in $(seq 1  ${ITR})
    do
        ${EXE} --k 1 --d 100 --rw=1 --ol=1  --input ~/data/${DATA[idx-1]}${GR} --ngpu=4 --hd=${HD[idx-1]} --n=400000
    done
done


# echo "-------------------------------------------------------online node2vec"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     ./node2vec --node2vec --ol=1  --bias=1  --d 100 --n=4000  --input ~/data/${DATA[idx-1]}${GR} 
# done


# echo "-------------------------------------------------------online sage 4k 25,10"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     ${EXE} --sage=1 --ol=1 --input ~/data/${DATA[idx-1]}${GR} --hd=${HD[idx-1]}
# done

# echo "-------------------------------------------------------online sample 400k 2,2"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     for i in $(seq 1  ${ITR})
#     do
#         ${EXE} --k 2 --d 2 --ol=1  --input ~/data/${DATA[idx-1]}${GR} --ngpu=4 --hd=${HD[idx-1]} --n=400000
#     done
# done


# echo "-------------------------------------------------------offline ppr 4k 0.15"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     ${EXE} --input ~/data/${DATA[idx-1]}${GR}  --hd=${HD[idx-1]} -bias=1 --rw=1 --ol=1 --n=4000 --k 1 --d 100  --tp=0.85 
# done


# echo "-------------------------------------------------------offline rw |V| 100. no weight"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     ${EXE} --rw=1 --k 1 --d 100 --ol=1 --bias=0  --full --input ~/data/${DATA[idx-1]}${GR}
# done


# echo "-------------------------------------------------------offline sample |V| 2,2"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     ${EXE} --rw=0 --k 2 --d 2 --ol=0 --input ~/data/${DATA[idx-1]}${GR}
# done

# echo "-------------------------------------------------------offline rw 4k 100"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     for ng in $(seq 1 4)
#     do
#         ${EXE} --rw=1 --k 1 --d 100 --ol=0  --input ~/data/${DATA[idx-1]}${GR} --ngpu=${ng} --hd=${HD[idx-1]} --n=40000
#     done
# done

# echo "-------------------------------------------------------offline  4k 10 10 "
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     for ng in $(seq 1 4)
#     do
#         ${EXE} --rw=0 --k 10 --d 2 --ol=0  --input ~/data/${DATA[idx-1]}${GR} --ngpu=${ng} --hd=${HD[idx-1]} --n=4000
#     done
# done


# ///////////////////
# |V| hard
# echo "-------------------------------------------------------offline rw |V| 100"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     ${EXE} --rw=1 --k 1 --d 100 --ol=0 --full --input ~/data/${DATA[idx-1]}${GR}
# done



