#!/bin/bash

ONLINE=false
BIAS=false
FULL=false
STATIC=false
BUFFER=false
for i in "$@"
do
case $i in
    -app=*|--app=*)
    APP="${i#*=}"
    shift # past argument=value
    # ;;
    # -s=*|--searchpath=*)
    # SEARCHPATH="${i#*=}"
    # shift # past argument=value
    # ;;
    # -l=*|--lib=*)
    # LIBPATH="${i#*=}"
    # shift # past argument=value
    ;;
    -online)
    ONLINE=true
    shift # past argument with no value
    ;;
    -bias)
    BIAS=true
    shift # past argument with no value
    ;;
    -full)
    FULL=true
    shift # past argument with no value
    ;;
    -static)
    STATIC=true
    shift # past argument with no value
    ;;
    -buffer)
    BUFFER=true
    shift # past argument with no value
    ;;
    *)
          # unknown option
    ;;
esac
done
# echo "APP  = ${APP}"
# echo "SEARCH PATH     = ${SEARCHPATH}"
# echo "LIBRARY PATH    = ${LIBPATH}"
# echo "DEFAULT         = ${DEFAULT}"
# echo "Number files in SEARCH PATH with EXTENSION:" $(ls -1 "${SEARCHPATH}"/*."${EXTENSION}" | wc -l)
# if [[ -n $1 ]]; then
#     echo "Last line of file specified as non-opt/last argument:"
#     tail -1 $1
# fi
echo ${BIN}
BIN="./bin/main"
if [ ${APP} = "node2vec" ] 
then
    if ${BIAS} 
    then
        BIN="./bin/node2vec"
    fi
fi
# echo ${BIN}
# if [  ${APP}="node2vec" ] && ${BIAS}  ; then
#     BIN="./bin/node2vec"
# else
#     BIN="./bin/main"
# fi

if ${BIAS} ; then
    BIN=${BIN}" -bias=1 "
else
    BIN=${BIN}" -bias=0 "
fi

if ${ONLINE} ; then
    BIN=${BIN}" -ol=1 "
else
    BIN=${BIN}" -ol=0 "
fi

if  ${FULL} ; then
    BIN=${BIN}" --full "
else
    BIN=${BIN}" --n 40000 "
fi
if  ${STATIC} ; then
    BIN=${BIN}" --static=1 "
else
    BIN=${BIN}" --static=0 "
fi
if  ${BUFFER} ; then
    BIN=${BIN}" --buffer=1 "
else
    BIN=${BIN}" --buffer=0 "
fi

DATA=(web-Google lj orkut arabic-2005 uk-2005  sk-2005 friendster) # uk-union rmat29 web-ClueWeb09) eu-2015-host-nat twitter-2010
HD=(0.25          0.5  1     0.25        0.25      0.5           1) # uk-union rmat29 web-ClueWeb09)
NV=(916428    4847571 3072627  39459923   22744077     50636151 124836180)
# HD=(4             2   1     4         4       2           1) # uk-union rmat29 web-ClueWeb09)

# DATA=( sk-2005 friendster) 
# HD=(   4  1 )
ITR=1
NG=4 #8

GR=".w.gr"
EXE="./bin/main" #main_degree
SG="--ngpu=1 --s"
RW="--deepwalk "
SP="--sage "
# BATCH="--n 40000 "
LOG_FILE="offline.csv"

# echo "-------------------------------------------------------offline rw 100  ${BATCH}" >> ${LOG_FILE}
echo "-------------------------------------------------------${APP} ${BIN} BIAS=${BIAS} ONLINE=${ONLINE} FULL=${FULL}------------"  
for idx in $(seq 1 ${#DATA[*]}) 
do
    for i in $(seq 1  ${ITR})
    do
        ${BIN} ${SG} --${APP} --input ~/data/${DATA[idx-1]}${GR} --hd=${HD[idx-1]} ${BATCH} # >> ${LOG_FILE} 
    done
done