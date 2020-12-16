DATA=(lj okt uk-2005 twitter-2010 sk-2005 friendster) # $orkut uk-union rmat29 web-ClueWeb09)  

ED=".w.edge"



# # for c-saw
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     ~/graph_project_start/tuple_text_to_binary_csr_mem/text_to_bin.bin   ~/data/${DATA[idx-1]}${ED}  0 0 40
# done

# for c-saw
for idx in $(seq 1 ${#DATA[*]}) 
do
    ~/sampling/KnightKing/build/bin/gconverter -i ~/data/${DATA[idx-1]}${ED} -o ./${DATA[idx-1]}.data -s weighted 
done