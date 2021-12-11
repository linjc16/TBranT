# SCIP solver

Set-up a desired installation path for SCIP / SoPlex (e.g., `/opt/scip`):
```
$ export SCIPOPTDIR='/opt/scip'
```

## SoPlex

SoPlex 4.0.1 (free for academic uses)

https://soplex.zib.de/download.php?fname=soplex-4.0.1.tgz

```
$ tar -xzf soplex-4.0.1.tgz
$ cd soplex-4.0.1/
$ mkdir build
$ cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$SCIPOPTDIR
$ make -C ./build -j 4
$ make -C ./build install
$ cd ..
```

# SCIP

SCIP 6.0.1 (free for academic uses)

https://scip.zib.de/download.php?fname=scip-6.0.1.tgz

```
$ tar -xzf scip-6.0.1.tgz
$ cd scip-6.0.1/
```


```
$ mkdir build
$ cmake -S . -B build -DSOPLEX_DIR=$SCIPOPTDIR -DCMAKE_INSTALL_PREFIX=$SCIPOPTDIR
$ make -C ./build -j 4
$ make -C ./build install
$ cd ..
```

For reference, original installation instructions [here](http://scip.zib.de/doc/html/CMAKE.php).

# Python dependencies

Recommended setup: conda + python 3

https://docs.conda.io/en/latest/miniconda.html

## Cython

Required to compile PySCIPOpt and PySVMRank
```
$ conda install cython
```

## PySCIPOpt

SCIP's python interface (modified version)

```
$ pip install git+https://github.com/ds4dm/PySCIPOpt.git@branch-search-trees
```

or 
```
$ cd PySCIPOpt
$ export SCIPOPTDIR='/opt/scip'
$ python setup.py install
```