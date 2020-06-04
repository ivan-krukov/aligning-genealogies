Pedigree format
===============

We are using the table format from the *BALSAC* (cite) project. The input is a table with at least 3
columns:

- Individual ID
- Paternal ID
- Maternal ID

In addition, a fourth column, indicating sex of the individual. By convention, ``1`` represents
males, ``2`` - females.

Example
-------

A simple 8 individual pedigree looks like this::

  individual    father  mother  sex
  1             0       0       1
  2             0       0       2
  3             0       0       2
  4             1       2       1
  5             1       2       2
  6             0       0       1
  7             4       3       2
  8             6       5       1

TODO: Visualize this
