###
 # @Description: 
 # @Date: 2020-11-25 16:50:34
 # @LastEditors: PengyuWang
 # @LastEditTime: 2021-01-11 16:30:59
 # @FilePath: /skywalker/result/knightking.sh
### 
DATA=(web-Google    lj    orkut      ) #   sk-2005 friendster) #  twitter-2010 uk-union rmat29 web-ClueWeb09)  
NV=(916428          4847571 3072627  39459923   22744077     50636151 124836180) #41652230

DIR="/home/pywang/sampling/nextdoor-experiments/NextDoor/src/apps/randomwalks/"
# DeepWalkSampling
# Node2VecSampling
# PPRSampling
# KHopSampling
echo "----------------------unbiased_walk  -------------------"
for idx in $(seq 1 ${#DATA[*]}) 
do
    echo "------------"${DATA[idx-1]}
    ${DIR}DeepWalkSampling  -g ~/data/${DATA[idx-1]}.data  -t edge-list -f binary -n 1 -k TransitParallel -l
done

# echo "----------------------unbiased_walk  -------------------"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     echo "------------"${DATA[idx-1]}
#     ${DIR}DeepWalkSampling  -g ~/data/${DATA[idx-1]}.data  -t edge-list -f binary -n 1 -k TransitParallel -l
# done
# echo "----------------------ppr unbiased -------------------"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     echo "------------"${DATA[idx-1]}
#     ${DIR}PPRSampling -g ~/data/${DATA[idx-1]}.data  -t edge-list -f binary -n 1 -k TransitParallel -l
# done
# echo "----------------------node2vec -------------------"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     echo "------------"${DATA[idx-1]}
#     ${DIR}Node2VecSampling -g ~/data/${DATA[idx-1]}.data  -t edge-list -f binary -n 1 -k TransitParallel -l
# done
# echo "----------------------kh -------------------"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     echo "------------"${DATA[idx-1]}
#     /home/pywang/sampling/nextdoor-experiments/NextDoor/src/apps/khop/KHopSampling -g ~/data/${DATA[idx-1]}.data  -t edge-list -f binary -n 1 -k TransitParallel -l
# done
