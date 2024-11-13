# setup

compile
```bash
module load mvapich2
mpic++ main.cpp -o main
```

run
```bash
mpirun -np 4 ./main -P 1 -N 10000
```
