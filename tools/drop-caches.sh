###
 # @Description: 
 # @Date: 2021-01-10 22:36:32
 # @LastEditors: PengyuWang
 # @LastEditTime: 2021-01-10 22:37:02
 # @FilePath: /skywalker/tools/drop-caches.sh
### 
# Mark it as executable using chmod a+x drop-caches
# Call it using sudo ./drop-caches
# If you place the script in /usr/local/bin you can call it using sudo drop-caches

#!/bin/bash
if [[ $(id -u) -ne 0 ]] ; then echo "Please run as root" ; exit 1 ; fi
sync; echo 1 > /proc/sys/vm/drop_caches
sync; echo 2 > /proc/sys/vm/drop_caches
sync; echo 3 > /proc/sys/vm/drop_caches