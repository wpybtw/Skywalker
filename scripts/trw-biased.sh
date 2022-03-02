###
 # @Description: 
 # @Date: 2021-10-21
 # @LastEditors: Xu Cheng
 # @FilePath: /ThunderRW
### 
DATA=(GG lj OK AB UK SK FS) # uk-union rmat29 web-ClueWeb09) eu-2015-host-nat twitter-2010


 echo "biased rw" >>"/home/xucheng/ThunderRW/biased3.csv"
 for idx in $(seq 1 ${#DATA[*]}) 
 do
     echo " result of ${DATA[idx-1]} " >>"/home/xucheng/ThunderRW/biased3.csv"
     ./build/random_walk/deepwalk.out -f sample_dataset/${DATA[idx-1]}/ -n 20  -ew -l 100 >>"/home/xucheng/ThunderRW/biased3.csv" 2>&1 
     echo " " >>"/home/xucheng/ThunderRW/biased3.csv"
 done
 
 echo " " >>"/home/xucheng/ThunderRW/biased3.csv"
 
 echo "biased ppr" >>"/home/xucheng/ThunderRW/biased3.csv"
 for idx in $(seq 1 ${#DATA[*]}) 
 do
     echo " result of ${DATA[idx-1]} " >>"/home/xucheng/ThunderRW/biased3.csv"
     ./build/random_walk/ppr.out -f sample_dataset/${DATA[idx-1]}/ -n 20 -sp 0.15 -em 1 -sm 2 -l 100  >>"/home/xucheng/ThunderRW/biased3.csv" 2>&1 
     echo " " >>"/home/xucheng/ThunderRW/biased3.csv"
 done
 
 echo " " >>"/home/xucheng/ThunderRW/biased3.csv"
 
 echo "biased node2vec" >>"/home/xucheng/ThunderRW/biased3.csv"
 for idx in $(seq 1 ${#DATA[*]}) 
 do
     echo " result of ${DATA[idx-1]} " >>"/home/xucheng/ThunderRW/biased3.csv"
      ./build/random_walk/node2vec.out -f sample_dataset/${DATA[idx-1]}/ -n 20 -ew -l 100 >>"/home/xucheng/ThunderRW/biased3.csv" 2>&1 
     echo " " >>"/home/xucheng/ThunderRW/biased3.csv"
 done
 
 
 


