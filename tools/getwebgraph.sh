###
 # @Description: 
 # @Date: 2021-01-04 22:39:00
 # @LastEditors: PengyuWang
 # @LastEditTime: 2021-01-04 22:52:19
 # @FilePath: /sampling/scripts/downloadgraph.sh
### 

cd ~/data


wget http://data.law.di.unimi.it/webdata/$1/$1.properties
wget http://data.law.di.unimi.it/webdata/$1/$1.graph

cd webgraph-big-3.5.1

java -cp "*" it.unimi.dsi.webgraph.ArcListASCIIGraph    ../$1 ../$1

cd ..
python2 ~/graph/gunrock/tools/associate_weights.py ~/data/$1

mv $1.random.weight.mtx $1.w.edge
~/graph/Galois/build/tools/graph-convert/graph-convert -edgelist2gr  ~/data/$1.w.edge ~/data/$1.w.gr -edgeType=uint32