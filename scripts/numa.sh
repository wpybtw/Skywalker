###
 # @Description: https://stackoverflow.com/questions/16056800/multi-gpu-programming-using-cuda-on-a-numa-machine
 # @Date: 2020-12-29 18:14:52
 # @LastEditors: PengyuWang
 # @LastEditTime: 2020-12-29 18:16:18
 # @FilePath: /sampling/scripts/numa.sh
### 
#!/bin/bash
#this script will output a listing of each GPU and it's CPU core affinity mask
file="/proc/driver/nvidia/gpus/0000:3d:00.0/information"
if [ ! -e $file ]; then
  echo "Unable to locate any GPUs!"
else
  gpu_num=0
  file="/proc/driver/nvidia/gpus/$gpu_num/information"
  if [ "-v" == "$1" ]; then echo "GPU:  CPU CORE AFFINITY MASK: PCI:"; fi
  while [ -e $file ]
  do
    line=`grep "Bus Location" $file | { read line; echo $line; }`
    pcibdf=${line:14}
    pcibd=${line:14:7}
    file2="/sys/class/pci_bus/$pcibd/cpuaffinity"
    read line2 < $file2
    if [ "-v" == "$1" ]; then
      echo " $gpu_num     $line2                  $pcibdf"
    else
      echo " $gpu_num     $line2 "
    fi
    gpu_num=`expr $gpu_num + 1`
    file="/proc/driver/nvidia/gpus/$gpu_num/information"
  done
fi