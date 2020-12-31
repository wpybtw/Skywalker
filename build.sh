###
 # @Description: https://stackoverflow.com/questions/21134120/how-to-turn-makefile-into-json-compilation-database
 # @Date: 2020-12-29 21:36:13
 # @LastEditors: PengyuWang
 # @LastEditTime: 2020-12-29 21:45:17
 # @FilePath: /sampling/src/build.sh
### 
make --always-make --dry-run \
 | grep -wE 'gcc|g++|nvcc' \
 | grep -w '\-c' \
 | jq -nR '[inputs|{directory:".", command:., file: match(" [^ ]+$").string[1:]}]' \
 > compile_commands.json