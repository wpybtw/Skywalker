
###
 # @Description: 
 # @Date: 2020-11-25 16:31:37
 # @LastEditors: PengyuWang
 # @LastEditTime: 2020-12-09 19:36:52
 # @FilePath: /sampling/scripts/csaw.sh
### 
cd ../others/C-SAW/non-stream

./streaming.bin  wg  ~/data/soc-LiveJournal1.txt_beg_pos.bin  ~/data/soc-LiveJournal1.txt_csr.bin  100 32 4000 1 2 2 1
./streaming.bin  wg  ~/data/lj.w.edge_beg_pos.bin  ~/data/lj.w.edge_beg_csr.bin  100 32 4000 1 1 100 1


./streaming.bin  wg  ~/data/orkut.w.edge_beg_pos.bin  ~/data/orkut.w.edge_csr.bin  100 32 4000 1 1 100 1
./streaming.bin  wg  ~/data/orkut.w.edge_beg_pos.bin  ~/data/orkut.w.edge_csr.bin  100 32 4000 1 2 2 1

