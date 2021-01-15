###
 # @Description: 
 # @Date: 2021-01-04 22:39:00
 # @LastEditors: PengyuWang
 # @LastEditTime: 2021-01-10 14:24:09
 # @FilePath: /skywalker/scripts/getsnapgraph.sh
### 

cd ~/data
wget http://data.law.di.unimi.it/webdata/$1/$1.properties
# mv $1.txt $1
# python2 ~/graph/gunrock/tools/associate_weights.py ~/data/$1

# mv $1.random.weight.mtx $1.w.edge
# ~/graph/Galois/build/tools/graph-convert/graph-convert -edgelist2gr  ~/data/$1.w.edge ~/data/$1.w.gr -edgeType=uint32