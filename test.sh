#!/bin/bash

length=(10 20 30 40 50 60 70 80 90 100 110 120)
spectrum=(64 100)

for spec in ${spectrum[@]}
do
    for len in ${length[@]}
    do
        echo "------------------------------------------------------------------------------------"
        echo "signal length (s): $len, sampling frequency (Hz): 1000, spectrum size (#): $spec"
        $(nvprof --print-gpu-trace ./tester $len 1000 $spec)
    done
done