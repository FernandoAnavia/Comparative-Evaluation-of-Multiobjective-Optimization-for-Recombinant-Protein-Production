# Comparative-Evaluation-of-Multiobjective-Optimization-for-Recombinant-Protein-Production

This repository contains the source code and run-level data used in the study:

**Comparative Evaluation of Evolutionary and Gradient-Based Multiobjective Optimization for Recombinant Protein Production**

The study compares eight configurations for maximizing protein productivity and process yield in a synthetic *Escherichia coli* consortium: NSGA-II, five MOEA/D decomposition variants (WS, TCH, PBI, ASF, and AASF), and weighted-sum FMINCON using 49 and 100 weights.

Thirty independent runs were performed for each configuration. Performance was evaluated using hypervolume, spacing, non-dominated output cardinality, runtime, and outer objective-function evaluations. The statistical analysis uses Kruskal–Wallis tests followed by Holm-adjusted Mann–Whitney U comparisons against NSGA-II, with Cliff’s delta as the effect-size measure. A reference-point sensitivity analysis is also included.

## Software

- Python 3.13.7
- pymoo 0.6.1.6
- SciPy 1.16.3
- MATLAB R2026a
- Optimization Toolbox 26.1
- irace 4.4.3
- R 4.6.0

## Authors

José Fernando González-Anavia, Adriana Lara López, and Ponciano Jorge Escamilla Ambrosio.
