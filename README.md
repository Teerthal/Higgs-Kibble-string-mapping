# Higgs-Kibble-string-mapping
A python code for mapping monopole-antimonpole pairs and strings in randomized Higgs field configurations using the Kibble Mechanism.
A high-level language like Python allows dynamic arrays and its efficient Numpy library allows querying t 
The code has been accelerated using the JIT(Just-In-Time compiler) decorators by NUMBA.
This can be almost trivially accomplished, as in the functions in the scripts here, by restricting the functions being called to numpy functions and using static arrays inside the desired routine.

## Example of a code that finds the monopole-antimonopole pair and tracks the string connecting them.
[box.pdf](https://github.com/Teerthal/Higgs-Kibble-string-mapping/files/12470258/box.pdf)

## Some Documentation for the search and tracking algorithms
[document.pdf](https://github.com/Teerthal/Higgs-Kibble-string-mapping/files/12470272/document.pdf)

### The documentation includes algorithm flowcharts for the attempted dumbbell collapse schemes.
The corresponding code files are in the topological_collapsing_scheme folder.
The code is written primarily in Julia since Julia was found to be far quicker than its Python counterpart.
The mathematical formulation and the algorithm need further refinement since the currently employed transformations that move the monopole-antimonopole pair along the string results in unwanted additional monopoles being created.
