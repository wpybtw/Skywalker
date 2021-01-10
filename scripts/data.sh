###
 # @Description: 
 # @Date: 2020-11-17 13:30:33
 # @LastEditors: PengyuWang
 # @LastEditTime: 2021-01-10 14:22:31
 # @FilePath: /skywalker/scripts/data.sh
### 
DATA=(    sk-2005 friendster) # $orkut uk-union rmat29 web-ClueWeb09)   twitter-2010
# lj orkut web-Google uk-2005

ED=".w.edge"

EL=".el"


# # for c-saw
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     ~/graph_project_start/tuple_text_to_binary_csr_mem/text_to_bin.bin   ~/data/${DATA[idx-1]}${ED}  0 0 40
# done


for idx in $(seq 1 ${#DATA[*]}) 
do
    ~/sampling/KnightKing/build/bin/gconverter -i ~/data/${DATA[idx-1]}${EL} -o ~/data/${DATA[idx-1]}.uw.data -s unweighted 
done