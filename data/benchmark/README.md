# some quick benchmarks

Time is measured in s/it, if not otherwise specified.


## Exact Diagionalization

* dataset 9-12, any dt:  4.09 s/it


## quimb MPS

* dataset 9-12, dt = 0.5

| bd | time (CPU) | time (torch GPU) |
| --- | --- | --- |
| 10 | 0.66 | 0.68 |
| 20 | 1.06 | 0.77 |
| 40 | 1.36 | 0.81 |
| 60 | 1.40 | 0.89 | 
| 80 | - | - |

Note: missing 80 bench because crops automatically to bd=64


## numpy MPS

* dataset 9-12, dt = 0.5

| bd | time (CPU) | time (jit CPU) | time (jit GPU) |
| --- | --- | --- | --- |
| 10 | 0.45 | 0.27 | 0.40 |
| 20 | 0.83 | 0.45 | 0.51 |
| 40 | 2.23 | 2.15 | 0.69 |
| 60 | 4.38 | 4.97 | 0.86 |
| 80 | 5.47 | 7.03 | 1.04 |