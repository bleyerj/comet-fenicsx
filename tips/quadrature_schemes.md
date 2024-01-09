---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Quadrature schemes

```{admonition} Objectives
:class: objectives

This code snippet shows the location of the quadrature points for different degrees, cell types and quadrature rules.
```

## Code

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import basix
from basix import CellType, QuadratureType

all_cell_types = [CellType.triangle, CellType.quadrilateral]
degrees = range(5)
quad_rules = {
    QuadratureType.Default: (degrees, all_cell_types),
    QuadratureType.gauss_jacobi: (degrees, all_cell_types),
    QuadratureType.gll: (degrees, [CellType.quadrilateral]),
    QuadratureType.xiao_gimbutas: (degrees[1:], [CellType.triangle]),
}


def plot_quad_rule(rule, color="C0"):
    degs, cell_types = quad_rules[rule]
    for deg in degs:
        plt.figure()
        for i, cell_type in enumerate(cell_types):
            no_subplot = len(cell_types) < 2
            points, weights = basix.make_quadrature(cell_type, deg, rule=rule)
            if no_subplot:
                ax = plt.gca()
            else:
                ax = plt.subplot(1, 2, i + 1)
            ax.margins(0.05)

            vertices = basix.geometry(cell_type)
            facets = basix.cell.sub_entity_connectivity(cell_type)[1]

            for f in facets:
                vert = vertices[f[0], :]
                ax.plot(vert[:, 0], vert[:, 1], "k")

            ax.scatter(points[:, 0], points[:, 1], 500 * weights, color=color)
            ax.set_aspect("equal")

        if no_subplot:
            plt.title(f"{rule.name} rule, degree $d={deg}$")
        else:
            plt.suptitle(f"{rule.name} rule, degree $d={deg}$", y=0.8)
        plt.show()
```

## Default quadrature rule

```{code-cell} ipython3
plot_quad_rule(QuadratureType.Default, color="C3")
```

## Gauss-Jacobi quadrature rule

```{code-cell} ipython3
plot_quad_rule(QuadratureType.gauss_jacobi, color="C2")
```

## Gauss-Legendre-Lobatto quadrature rule

```{code-cell} ipython3
plot_quad_rule(QuadratureType.gll, color="C1")
```

## Xiao-Gimbutas quadrature rule

```{code-cell} ipython3
plot_quad_rule(QuadratureType.xiao_gimbutas, color="C0")
```
