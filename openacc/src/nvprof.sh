#!/bin/sh
export PMI_NO_FORK=1
LOG=profile_$(hostname).nvp
setstripe -c 1 $LOG
exec nvprof -o $LOG $*

