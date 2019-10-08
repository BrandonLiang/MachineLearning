#!/bin/bash -e                                                                                                                                                                 
SCRIPT=$(greadlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)
APP_HOME=$SCRIPT_DIR/..
SRC=${APP_HOME}/src

FILEPATH="$1"
ROWS=$2 # number of rows to show
SHOW_ALL=$3 # False if yes, True if not

python ${SRC}\/spark_df.py "$FILEPATH" $ROWS $SHOW_ALL
