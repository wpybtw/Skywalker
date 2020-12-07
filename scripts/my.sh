cd ../src
###
 # @Description: 
 # @Date: 2020-11-25 16:38:51
 # @LastEditors: PengyuWang
 # @LastEditTime: 2020-12-06 17:45:38
 # @FilePath: /sampling/scripts/my.sh
### 

./main_degree --rw=0 --k 1 --d 100 --ol=1 --input ~/data/lj.w.gr
./main_degree --rw=0 --k 1 --d 100 --ol=1 --input ~/data/orkut.w.gr

./main_degree --rw=0 --k 2 --d 2 --ol=1 --input ~/data/lj.w.gr
./main_degree --rw=0 --k 2 --d 2 --ol=1 --input ~/data/orkut.w.gr

./main --rw=0 --k 2 --d 2 --ol=1 --randomweight=1 --weightrange=2 --input ~/data/lj.w.gr
./main --rw=0 --k 2 --d 2 --ol=1 --randomweight=1 --weightrange=2 --input ~/data/orkut.w.gr


./main --rw=0 --k 1 --d 100 --ol=1 --randomweight=1 --weightrange=2 --input ~/data/lj.w.gr
./main --rw=0 --k 1 --d 100 --ol=1 --randomweight=1 --weightrange=2 --input ~/data/orkut.w.gr

./main --rw=0 --k 1 --d 100 --ol=1  --input ~/data/lj.w.gr
./main --rw=0 --k 1 --d 100 --ol=1  --input ~/data/orkut.w.gr
./main --rw=1 --k 1 --d 100 --ol=1  --input ~/data/lj.w.gr
./main --rw=1 --k 1 --d 100 --ol=1  --input ~/data/orkut.w.gr

./main_degree --rw=0 --k 1 --d 100 --ol=1  --input ~/data/lj.w.gr
./main_degree --rw=0 --k 1 --d 100 --ol=1  --input ~/data/orkut.w.gr
./main_degree --rw=1 --k 1 --d 100 --ol=1  --input ~/data/lj.w.gr
./main_degree --rw=1 --k 1 --d 100 --ol=1  --input ~/data/orkut.w.gr



./main  --rw=1 --ol=0 --k 1 --d 100  --input ~/data/lj.w.gr
./main  --rw=1 --ol=0 --k 1 --d 100  --input ~/data/orkut.w.gr

./main  --rw=1 --ol=0 --n=4847571 --k 1 --d 100  --input ~/data/lj.w.gr
./main  --rw=1 --ol=0 --n=3072627 --k 1 --d 100  --input ~/data/orkut.w.gr