#!/bin/bash -e                                                                                                                                                                 
SCRIPT=$(greadlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)
APP_HOME=$SCRIPT_DIR/..
#SRC=${APP_HOME}/src
SRC=/Users/brandonliang/src/MachineLearning/src

FILEPATH=$1
ROWS=$2

python ${SRC}\/pandas_df.py $FILEPATH $ROWS
