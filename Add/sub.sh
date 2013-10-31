#!/bin/bash

qsub -cwd -P test_project -pe mpich 2 -hard -q nvidia_480.q testrun.sh