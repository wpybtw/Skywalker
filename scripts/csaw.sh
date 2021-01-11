
###
 # @Description: 
 # @Date: 2020-11-25 16:31:37
 # @LastEditors: PengyuWang
 # @LastEditTime: 2021-01-11 11:31:12
 # @FilePath: /skywalker/scripts/csaw.sh
### 
DATA=( web-Google    lj    orkut   arabic-2005   uk-2005    sk-2005 friendster) # 
# DATA=( uk-2005   sk-2005 friendster)
cd /home/pywang/sampling/C-SAW/non-stream

# ./sampling.bin  wg  ~/data/soc-LiveJournal1.txt_beg_pos.bin  ~/data/soc-LiveJournal1.txt_csr.bin  100 32 4000 1 2 2 1
# ./sampling.bin  wg  ~/data/lj.w.edge_beg_pos.bin  ~/data/lj.w.edge_csr.bin  100 32 4000 1 1 100 1
# ./sampling.bin  wg  ~/data/lj.w.edge_beg_pos.bin  ~/data/lj.w.edge_csr.bin  100 32 4000 1 2 2 1

# ./sampling.bin  wg  ~/data/orkut.w.edge_beg_pos.bin  ~/data/orkut.w.edge_csr.bin  100 32 4000 1 1 100 1
# ./sampling.bin  wg  ~/data/orkut.w.edge_beg_pos.bin  ~/data/orkut.w.edge_csr.bin  100 32 4000 1 2 2 1

# ./sampling.bin  wg  ~/data/uk-2005.w.edge_beg_pos.bin  ~/data/uk-2005.w.edge_csr.bin  100 32 4000 1 1 100 1
# ./sampling.bin  wg  ~/data/uk-2005.w.edge_beg_pos.bin  ~/data/uk-2005.w.edge_csr.bin  100 32 4000 1 2 2 1

# echo "----------------------biased walk 4k 64-------------------"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     echo ${DATA[idx-1]}
#     /home/pywang/sampling/C-SAW/non-stream/sampling.bin  wg  ~/data/${DATA[idx-1]}.w.edge_beg_pos.bin   ~/data/${DATA[idx-1]}.w.edge_csr.bin 100 32 4000 1 1 100 1
# done

echo "----------------------sampling biased 4k 2 2 64-------------------"
for idx in $(seq 1 ${#DATA[*]}) 
do
    echo ${DATA[idx-1]}
    /home/pywang/sampling/C-SAW/non-stream/sampling.bin  wg  ~/data/${DATA[idx-1]}.w.edge_beg_pos.bin   ~/data/${DATA[idx-1]}.w.edge_csr.bin 100 32 4000 1 20 2 1
done

# echo "----------------------biased walk 4k 64-------------------"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     echo ${DATA[idx-1]}
#     /home/pywang/sampling/C-SAW/streaming/streaming.bin  wg  ~/data/${DATA[idx-1]}.w.edge_beg_pos.bin   ~/data/${DATA[idx-1]}.w.edge_csr.bin 100 32 4000 1 1 100 1
# done

# echo "----------------------sampling biased 4k 20 2 64-------------------"
# for idx in $(seq 1 ${#DATA[*]}) 
# do
#     echo ${DATA[idx-1]}
#     /home/pywang/sampling/C-SAW/streaming/streaming.bin  wg  ~/data/${DATA[idx-1]}.w.edge_beg_pos.bin   ~/data/${DATA[idx-1]}.w.edge_csr.bin 100 32 4000 1 20 2 1
# done