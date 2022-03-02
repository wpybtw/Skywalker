###
 # @Description: 
 # @Date: 2020-11-25 16:50:34
 # @LastEditors: PengyuWang
 # @LastEditTime: 2021-01-11 16:30:59
 # @FilePath: /skywalker/result/knightking.sh
### 
DATA=( web-Google    lj    orkut      uk-2005 arabic-2005   sk-2005 friendster) #  twitter-2010 uk-union rmat29 web-ClueWeb09)  
NV=(916428          4847571 3072627  39459923   22744077     50636151 124836180) #41652230
# cd ../KnightKing/build
# DATA=(   lj        ) #  twitter-2010 uk-union rmat29 web-ClueWeb09)  
# NV=(  4847571         ) #41652230
# DATA=(  web-Google  orkut      arabic-2005   ) #  twitter-2010 uk-union rmat29 web-ClueWeb09)  
# NV=(916428  3072627     22744077     ) #41652230

# echo "----------------------unbiased 4k degree-------------------"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     echo ${DATA[idx-1]}
#     ~/sampling/KnightKing/build/bin/deepwalk  -w 40000  -l 100 -s unweighted -g ~/data/${DATA[idx-1]}.uw.data -v ${NV[idx-1]}
# done
# echo "----------------------unbiased node2vec-------------------"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     echo ${DATA[idx-1]}
#     ~/sampling/KnightKing/build/bin/node2vec  -w 40000  -l 100 -s unweighted  -p 2.0 -q 0.5 -g ~/data/${DATA[idx-1]}.uw.data -v ${NV[idx-1]} 
# done

# echo "----------------------ppr unbiased 40k 64-------------------"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     echo ${DATA[idx-1]}
#     ~/sampling/KnightKing/build/bin/ppr  -w 40000  -s unweighted  -t 0.15 -v  ${NV[idx-1]} -g ~/data/${DATA[idx-1]}.uw.data 
# done


# echo "----------------------simple_walk 4k-------------------"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     echo ${DATA[idx-1]}
#     ~/sampling/KnightKing/build/bin/simple_walk -g ~/sampling/KnightKing/build/${DATA[idx-1]}.data -v ${NV[idx-1]} -w 4000  -l 100
# done

# echo "----------------------online 40k-------------------"
# echo "----------------------biased_walk 4k degree-------------------"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     echo ${DATA[idx-1]}
#     ~/sampling/KnightKing/build/bin/biased_walk  -w 40000  -l 100 -g ~/data/${DATA[idx-1]}.data -v ${NV[idx-1]}
# done
# echo "----------------------ppr biased 4k 64-------------------"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     echo ${DATA[idx-1]}
#     ~/sampling/KnightKing/build/bin/ppr  -s weighted  -t 0.15  -w 40000  -g ~/data/${DATA[idx-1]}.data -v ${NV[idx-1]}
# done
# echo "----------------------node2vec-------------------"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     echo ${DATA[idx-1]}
#     ~/sampling/KnightKing/build/bin/node2vec  -w 40000  -l 100 -s weighted  -p 2.0 -q 0.5 -g ~/data/${DATA[idx-1]}.data -v ${NV[idx-1]} 
# done

# echo "----------------------biased_walk 40k degree-------------------"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     echo ${DATA[idx-1]}
#     ~/sampling/KnightKing/build/bin/biased_walk  -w 40000  -l 100 -g ~/data/${DATA[idx-1]}.data -v ${NV[idx-1]}
# done



# echo "----------------------ppr biased 40k 64-------------------"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     echo ${DATA[idx-1]}
#     ~/sampling/KnightKing/build/bin/ppr  -s weighted  -t 0.15  -w 40000  -g ~/data/${DATA[idx-1]}.data -v ${NV[idx-1]}
# done


# echo "----------------------biased_walk 40k degree-------------------"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     echo ${DATA[idx-1]}
#     ~/sampling/KnightKing/build/bin/biased_walk -w 40000 -g ~/data/${DATA[idx-1]}.data -v ${NV[idx-1]}   -l 100
# done
# -w ${NV[idx-1]}

echo "----------------------biased node2vec-------------------"
for idx in $(seq 1 ${#DATA[*]}) 
do
    echo ${DATA[idx-1]}
    ~/sampling/KnightKing/build/bin/node2vec  -w 40000  -l 100 -s weighted  -p 2.0 -q 0.5 -g ~/data/${DATA[idx-1]}.data -v ${NV[idx-1]} 
done
