#!/bin/bash
# rm -f target/result.csv
for N in 96
do
    for R in "0.05" "0.1" "0.15" "0.2" "0.25"
    do
        for K in 1 2 3 4 5 6 7 8 9 10
        do
            python3 -W ignore ./src/main.py -n $N -r $R -k $K -o machine >> target/result-seed.csv
        done
    done
    for T in random popular consensus controversial
    do
        for S in 1 2 3 4 5
        do
            for R in 0.05 0.1 0.15 0.2 0.25
            do
                for K in 1 2 3 4 5 6 7 8 9 10
                do
                    python3 -W ignore ./src/main.py -n $N -t $T -s $S -r $R -k $K -o machine >> target/result-seed.csv
                done
            done
        done
    done
done