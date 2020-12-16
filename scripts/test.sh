DATA=(lj orkut  uk-2005 twitter-2010 sk-2005 friendster) # uk-union rmat29 web-ClueWeb09)
HD=(4 1  4 1 4 1) # uk-union rmat29 web-ClueWeb09)
###
 # @Description: 
 # @Date: 2020-11-17 13:39:45
 # @LastEditors: PengyuWang
 # @LastEditTime: 2020-12-09 20:21:55
 # @FilePath: /sampling/scripts/test.sh
### 
GR=".w.gr"

# --randomweight=1 --weightrange=2 



# echo "-------------------------------------------------------online rw 4k 100. using 0-64 weight"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     ./main --rw=0 --k 1 --d 100 --ol=1 --input ~/data/${DATA[idx-1]}${GR}
# done

# echo "-------------------------------------------------------offline rw |V| 100. no weight"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     ./main --rw=1 --k 1 --d 100 --ol=1 --bias=0  --full --input ~/data/${DATA[idx-1]}${GR}
# done

# echo "-------------------------------------------------------unbias rw 4000 100. no weight"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     ./main --rw=1 --k 1 --d 100 --ol=1 --bias=0  --input ~/data/${DATA[idx-1]}${GR}
# done

# |V| hard
# echo "-------------------------------------------------------offline rw |V| 100. using 0-64 weight"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     ./main --rw=1 --k 1 --d 100 --ol=0 --full --input ~/data/${DATA[idx-1]}${GR}
# done



# echo "-------------------------------------------------------online node2vec. using 0-64 weight"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     ./node2vec --node2vec --ol=1  --bias=1  --d 100 --n=4000  --input ~/data/${DATA[idx-1]}${GR} 
# done


# echo "-------------------------------------------------------online sage 4k 25,10. using 0-64 weight"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     ./main --sage=1 --ol=1 --input ~/data/${DATA[idx-1]}${GR} --hd=${HD[idx-1]}
# done

# echo "-------------------------------------------------------online sample 4k 2,2. using 0-64 weight"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     ./main --k 2 --d 2 --ol=1 --input ~/data/${DATA[idx-1]}${GR}  --hd=${HD[idx-1]}
# done

echo "-------------------------------------------------------online sample 4k 2,2. using degree"
for idx in $(seq 1 ${#DATA[*]}) 
do
    ./main_degree --k 2 --d 2 --ol=1 --input ~/data/${DATA[idx-1]}${GR}  --hd=${HD[idx-1]}
done


# echo "-------------------------------------------------------offline sample |V| 2,2. using 0-64 weight"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     ./main --rw=0 --k 2 --d 2 --ol=0 --input ~/data/${DATA[idx-1]}${GR}
# done