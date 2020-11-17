DATA=(orkut  uk-2005 twitter-2010 sk-2005   friendster)# uk-union rmat29 web-ClueWeb09)
ED=".w.edge"




for idx in $(seq 1 ${#DATA[*]}) 
do
    ~/graph_project_start/tuple_text_to_binary_csr_mem/text_to_bin.bin   ~/data/${DATA[idx-1]}${ED}  0 0 40
done