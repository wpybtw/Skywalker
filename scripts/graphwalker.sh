###
 # @Description: 
 # @Date: 2021-01-07 19:01:38
 # @LastEditors: PengyuWang
 # @LastEditTime: 2021-01-10 18:34:44
 # @FilePath: /skywalker/scripts/graphwalker.sh
### 
DATA=( lj.w.edge     arabic-2005.w.edge     uk-2005.w.edge  sk-2005 friendster.w.edge) # uk-union rmat29 web-ClueWeb09) eu-2015-host-nat twitter-2010
NV=(  4847571         22744077             39459923   50636151    124836180)

# DATA=( uk-2005.w.edge) 
# NV=(  39459923 )

# grep  "00_runtime\|g_loadSubGraph:\|file:"

ITR=1

ED=".w.edge"
EXE="./bin/apps/rwdomination" #main_degree
DIR="/home/pywang/sampling/GraphWalker"

cd $DIR
# ${EXE} file ~/data/${DATA[idx-1]}${ED} firstsource 0 numsources 400000 walkspersource 1 maxwalklength 100 prob 0.0 L 100 N 4847571 
echo "-------------------------------------------------------unbias rw 40000 100"
for idx in $(seq 1 ${#DATA[*]}) 
do
    ./bin/apps/rawrandomwalks file ~/data/${DATA[idx-1]} R 40000 L 100 N  ${NV[idx-1]}
done

# echo "-------------------------------------------------------unbias ppr 40000 100"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     ./bin/apps/msppr file ~/data/${DATA[idx-1]} firstsource 0 numsources 40000 walkspersource 1 maxwalklength 100 prob 0.15
# done


