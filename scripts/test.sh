DATA=(lj orkut  uk-2005 twitter-2010 sk-2005 friendster) # uk-union rmat29 web-ClueWeb09)
GR=".w.gr"

# --randomweight=1 --weightrange=2 

echo "\n-------------------------------------------------------online sample 4k 2,2. using 0-64 weight"
for idx in $(seq 1 ${#DATA[*]}) 
do
    ./main --rw=0 --k 2 --d 2 --ol=1 --input ~/data/${DATA[idx-1]}${GR}
done

echo "\n-------------------------------------------------------online rw 4k 100. using 0-64 weight"
for idx in $(seq 1 ${#DATA[*]}) 
do
    ./main --rw=0 --k 1 --d 100 --ol=1 --input ~/data/${DATA[idx-1]}${GR}
done

echo "\n-------------------------------------------------------offline rw |V| 100. using 0-64 weight"
for idx in $(seq 1 ${#DATA[*]}) 
do
    ./main --rw=1 --k 1 --d 100 --ol=0 --input ~/data/${DATA[idx-1]}${GR}
done

echo "\n-------------------------------------------------------offline sample |V| 2,2. using 0-64 weight"
for idx in $(seq 1 ${#DATA[*]}) 
do
    ./main --rw=0 --k 2 --d 2 --ol=0 --input ~/data/${DATA[idx-1]}${GR}
done