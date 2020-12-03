cd ../src

./main_gbuffer lj ~/data/soc-LiveJournal1.txt_beg_pos.bin  ~/data/soc-LiveJournal1.txt_csr.bin  32 4000 1 1 100 1
./main_gbuffer lj ~/data/soc-LiveJournal1.txt_beg_pos.bin  ~/data/soc-LiveJournal1.txt_csr.bin  32 4000 1 2 2 1


./main_gbuffer lj ~/data/orkut.w.edge_beg_pos.bin  ~/data/orkut.w.edge_csr.bin 32 4000 1 1 100 1
./main_gbuffer lj ~/data/orkut.w.edge_beg_pos.bin  ~/data/orkut.w.edge_csr.bin  32 4000 1 2 2 1




./sample_rw2 lj ~/data/orkut.w.edge_beg_pos.bin  ~/data/orkut.w.edge_csr.bin  32 3072627 1 1 100  1

./sample_rw2 lj ~/data/soc-LiveJournal1.txt_beg_pos.bin  ~/data/soc-LiveJournal1.txt_csr.bin  32  4847571 1 1 100 1