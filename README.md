# fair-factorization-machines-recommendation-comparison

This repository holds the code and additional artifacts for a thesis that will compare multiple methods of recommendation with fairness considuration.

## High level diagram

![High level view](images/Architecture.jpg)

## Required packages

- pandas
- lightfm
- numpy
- matplotlib

> [!NOTE]  
> Can be installed using pip

## Important links

[MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/)

[LightFM Dataset](https://making.lyst.com/lightfm/docs/lightfm.data.html)

[LightFM Dataset example](https://making.lyst.com/lightfm/docs/examples/dataset.html)

## How to run

```
python3 -W ignore ./src/main.py -h
```

## Helper run script

Helper run script will run all parameters variations and produce all results in target/result.csv.
To run it:

```
src/run.sh
```
