###
 # @Description: 
 # @Date: 2021-10-21
 # @LastEditors: Xu Cheng
 # @FilePath: /ThunderRW
### 
DATA=(GG lj OK AB UK SK FS) # uk-union rmat29 web-ClueWeb09) eu-2015-host-nat twitter-2010

#
# echo "unbiased rw" >>"/home/xucheng/ThunderRW/unbiased.csv"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     echo " result of ${DATA[idx-1]} " >>"/home/xucheng/ThunderRW/unbiased.csv"
#     ./build/random_walk/deepwalk.out -f sample_dataset/${DATA[idx-1]}/ -n 20  -em 0 -sm 0 -l 100 >>"/home/xucheng/ThunderRW/unbiased.csv" 2>&1 
#     echo " " >>"/home/xucheng/ThunderRW/unbiased.csv"
# done
# 
# echo " " >>"/home/xucheng/ThunderRW/unbiased.csv"
# 
# echo "unbiased ppr" >>"/home/xucheng/ThunderRW/unbiased.csv"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     echo " result of ${DATA[idx-1]} " >>"/home/xucheng/ThunderRW/unbiased.csv"
#     ./build/random_walk/ppr.out -f sample_dataset/${DATA[idx-1]}/ -n 20 -sp 0.15 -em 0 -sm 0  >>"/home/xucheng/ThunderRW/unbiased.csv" 2>&1 
#     echo " " >>"/home/xucheng/ThunderRW/unbiased.csv"
# done
# 
# echo " " >>"/home/xucheng/ThunderRW/unbiased.csv"
# 
# echo "unbiased node2vec" >>"/home/xucheng/ThunderRW/unbiased.csv"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     echo " result of ${DATA[idx-1]} " >>"/home/xucheng/ThunderRW/unbiased.csv"
#      ./build/random_walk/node2vec.out -f sample_dataset/${DATA[idx-1]}/ -n 20 -em 0 -sm 0 >>"/home/xucheng/ThunderRW/unbiased.csv" 2>&1 
#     echo " " >>"/home/xucheng/ThunderRW/unbiased.csv"
# done
# 
 
 echo "unbiased rw" >>"/home/xucheng/ThunderRW/unbiased40.csv"
 for idx in $(seq 1 ${#DATA[*]}) 
 do
     echo " result of ${DATA[idx-1]} " >>"/home/xucheng/ThunderRW/unbiased40.csv"
     ./build/random_walk/deepwalk.out -f sample_dataset/${DATA[idx-1]}/ -n 40  -em 0 -sm 0 -l 100 >>"/home/xucheng/ThunderRW/unbiased40.csv" 2>&1 
     echo " " >>"/home/xucheng/ThunderRW/unbiased40.csv"
 done
 
 
 
 


