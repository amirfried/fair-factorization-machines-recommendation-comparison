#!/bin/bash
rm -f target/result.csv
for R in 0.1 0.2 0.3 0.4 0.5
do
    for K in 1 2 3 4 5 6 7 8 9 10
    do
        python3 -W ignore ./src/main.py -r $R -k $K -o machine >> target/result.csv
    done
done
for T in random popular consensus controversial
do
    for S in 1 2 3 4 5
    do
        for R in 0.1 0.2 0.3 0.4 0.5
        do
            for K in 1 2 3 4 5 6 7 8 9 10
            do
                python3 -W ignore ./src/main.py -t $T -s $S -r $R -k $K -o machine >> target/result.csv
            done
        done
    done
done
