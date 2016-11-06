#!/usr/bin/env bash
module load mpi/openmpi-x86_64
mpiexec -n 2 python main.py
