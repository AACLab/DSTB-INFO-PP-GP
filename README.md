# DSTB-INFO-PP-GP
Distributed Informative Path Planning by Gaussian Processes

These codes were written in Julia 1.8.5.

There were two scenarios including 10 robots in 200[m]x200[m] and 25 robots in 400[m]x400[m].

Run "sim10robots200.jl" for the first scenario and "sim25robots400.jl" for the second scenario.

We accelerate the computation by distributed computing, please define the number of workers appropriate to your computational resources by variable "nP" in the above files.
