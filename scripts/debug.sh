cuobjdump -sass ./sampler_high_degree.cubin >dump.txt

nvdisasm ./sampler_high_degree.cubin -g > dump.disasm

cuda-gdb ./main_gbuffer lj ~/data/orkut.w.edge_beg_pos.bin  ~/data/orkut.w.edge_csr.bin  100 32 1 1 2 2 1

set cuda memcheck  on
r   lj ~/data/orkut.w.edge_beg_pos.bin  ~/data/orkut.w.edge_csr.bin  100 32 10 1 2 2 1