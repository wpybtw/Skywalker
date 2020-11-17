DATA=(orkut  uk-2005 twitter-2010 sk-2005   friendster) # uk-union rmat29 web-ClueWeb09)
ED=".w.edge"

for idx in $(seq 1 ${#DATA[*]}) 
do
    ./src/main lj ~/data/${DATA[idx-1]}${ED}_beg_pos.bin  ~/data/${DATA[idx-1]}${ED}_csr.bin  100 32 4000 1 2 2 1
done