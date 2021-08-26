# Skywalker

This is the repo for Skywalker, an Efficient Alias-method-based Graph Sampling and Random Walk on GPUs.

## Setup
```
git clone https://github.com/wpybtw/skywalker_artifact --recursive
```

Note that Cmake is not correctly setted yet. We use cmake to build glfags and then make. 
```
cd build
cmake ..
make -j
cd ..
make
```

## Dataset
Skywalker uses [Galios](https://iss.oden.utexas.edu/?p=projects/galois) graph format (.gr) as the input. Other formats like Edgelist (form [SNAP](http://snap.stanford.edu/data/index.html)) or Matrix Market can be transformed into it with GALOIS' graph-convert tool. Compressed graphs like [Webgraph](http://law.di.unimi.it/datasets.php) need to be uncompressed first.

Here is an example:
```
wget http://snap.stanford.edu/data/wiki-Vote.txt.gz
gzip -d wiki-Vote.txt.gz
$GALOIS_PATH/build/tools/graph-convert/graph-convert -edgelist2gr  ~/data/wiki-Vote.txt  ~/data/wiki-Vote.gr
```
## Running
Please run scripts in ./scripts for testing.
