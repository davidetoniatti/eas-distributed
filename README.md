# eas-distributed

This repository contains a distributed implementation of the **Expand-and-Sparsify (EaS)** algorithm for non-parametric classification, as proposed by Kaushik Sinha in *“Non-parametric classification via expand-and-sparsify representation”* (NeurIPS 2024) ([OpenReview link](https://openreview.net/forum?id=0d50Il6enG)).

---

## Overview

This project provides a **distributed implementation** of the EaS algorithm described in the paper (Algorithm 1), using the **Julia programming language**.

The implementation uses the `Distributed` and `DistributedArrays` libraries. In particular, the training phase is distributed across
multiple worker processes, where each worker is responsible for computing the weights with respect to a partition of the training set.

---

## Repository Structure

- `src/` – Julia implementations of the algorithm.
- `notebooks/` – Interactive notebook to test and visualize results.
- `test/` – Test suite for validating the implementation.
- `imgs/` – Images for documentation and notebooks.
- `Project.toml` – Julia project configuration file.

---

## Installation and Usage

> Coming soon...

---

## Reference

The implemented algorithm is based on the following paper:

**“Non-parametric classification via expand-and-sparsify representation”**  
*Kaushik Sinha*, NeurIPS 2024.  
[OpenReview link](https://openreview.net/forum?id=0d50Il6enG)

---

## Contributing

Contributions are welcome! If you have ideas for optimizations, new features, or documentation improvements, please open an issue or submit a pull request.  
Make sure that existing tests pass and add new tests for any added features.

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

