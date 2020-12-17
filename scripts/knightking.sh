DATA=(lj ok uk-2005 twitter-2010 sk-2005 friendster) # $orkut uk-union rmat29 web-ClueWeb09)  
NV=(4847571 3072627 39459923 41652230 50636151 124836180)
# cd ../KnightKing/build


# ./bin/biased_walk -g ./lj.data -v 4847571 -w 4000  -l 100 
# ./bin/biased_walk -g ./ok.data -v 3072627 -w 4000  -l 100  


# ./bin/biased_walk -g ./lj.data -v 4847571 -w 4847571  -l 100
# ./bin/biased_walk -g ./ok.data -v 3072627 -w 3072627  -l 100



# ./bin/simple_walk -g ./lj.data -v 4847571 -w  4847571  -l 100
# ./bin/simple_walk -g ./ok.data -v 3072627 -w 3072627  -l 100

# echo "----------------------simple_walk 4k-------------------"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     echo ${DATA[idx-1]}
#     ~/sampling/KnightKing/build/bin/simple_walk -g ~/sampling/KnightKing/build/${DATA[idx-1]}.data -v ${NV[idx-1]} -w 4000  -l 100
# done


# echo "----------------------biased_walk 4k degree-------------------"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     echo ${DATA[idx-1]}
#     ~/sampling/KnightKing/build/bin/biased_walk -g ~/sampling/KnightKing/build/${DATA[idx-1]}.data -v ${NV[idx-1]} -w 4000  -l 100
# done



echo "----------------------ppr biased 4k 64-------------------"
for idx in $(seq 1 ${#DATA[*]}) 
do
    echo ${DATA[idx-1]}
    ~/sampling/KnightKing/build/bin/ppr -g ~/sampling/KnightKing/build/${DATA[idx-1]}.data -v ${NV[idx-1]} -w 4000  -s weighted  -t 0.15
done
# echo "----------------------ppr unbiased 4k -------------------"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     echo ${DATA[idx-1]}
#     echo ~/sampling/KnightKing/build/${DATA[idx-1]}.data
#     ~/sampling/KnightKing/build/bin/ppr -g ~/sampling/KnightKing/build/${DATA[idx-1]}.data -v ${NV[idx-1]} -w 4000 -s unweighted  -t 0.75
# done
# echo "----------------------biased_walk V degree-------------------"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     echo ${DATA[idx-1]}
#     ~/sampling/KnightKing/build/bin/biased_walk -g ~/sampling/KnightKing/build/${DATA[idx-1]}.data -v ${NV[idx-1]} -w ${NV[idx-1]}  -l 100
# done
# echo "----------------------node2vec-------------------"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     echo ${DATA[idx-1]}
#     ~/sampling/KnightKing/build/bin/node2vec -g ~/sampling/KnightKing/build/${DATA[idx-1]}.data -v ${NV[idx-1]} -w 4000  -l 100 -s weighted  -p 2.0 -q 0.5
# done

