libgsgp: Geometric Symmetric Genetic Programming for Symbolic Regression
========================================================================

This project was created to support the experimental component of the paper:

G. Dick (2015) "Improving Geometric Semantic Genetic Programming with
Safe Tree Initialisation" in EuroGP 2015, Lecture Notes in Computer
Science 9025, pp. 28-40

If you use this project in your work, then a reference to the above
work would be greatly appreciated.

It is a relatively straightforward implementation of GSGP with the
following properties:

* the use of interval arithmetic to guide the construction of parse trees
* least squares estimation of crossover and mutation coefficients
* memoization of execution to improve performance

Installation
------------

This project should install on any platform supported by GCC. A
Makefile is included to build the base library and demo application. A
call to:
  make
should compile the project without any problems, and the resulting
binaries should appear in the dist directory.

Documentation
-------------

Documentation for the project can be found in the file doc.txt

License
-------

See [LICENSE](LICENSE) file.

Support
-------

Any questions or comments should be directed to Grant Dick
(grant.dick@otago.ac.nz)
