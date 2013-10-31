#!/bin/bash
#
#$ -cwd
#$ -j y
#$ -P test_project
#$ -m n
#$ -pe mpich 1
#$ -hard
#$ -S /bin/bash
#$ -q all.q
#
./a.out