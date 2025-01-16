---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# A module for rotation parametrization

```{admonition} Objectives
:class: objectives

This document describes the implementation of generic vectorial parametrization of rotation matrices,  essentially following  {cite:p}`bauchau2003vectorial`. This module is used in the [Nonlinear beam model in finite rotations](finite_rotation_nonlinear_beam.ipynb) tour.
```

## Implementation aspects

The module provides the `Skew` function, mapping a 3D vector to the corresponding skew-symmetric matrix.

An abstract class handles the generic implementation of rotation parametrization based on the corresponding parametrization of the rotation angle $p(\varphi)$. Considering a generic rotation vector which we denote by $\boldsymbol{\theta}$,  {cite:p}`bauchau2003vectorial` works with $\varphi = \|\boldsymbol{\theta}\|$ and the unit-norm vector $\boldsymbol{u} = \boldsymbol{\theta}/\varphi$. Note that the involved expressions are usually ill-defined when $\varphi \to 0$. For this reason, the numerical implementation makes use of a regularized expression for the norm:
\begin{equation}
\varphi = \sqrt{\boldsymbol{\theta}^2 + \varepsilon}
\end{equation}
with ${\varepsilon}=$ DOLFIN_EPS in practice.

The rotation parameter vector $\boldsymbol{p}$ from {cite:p}`bauchau2003vectorial` is given by the `rotation_parameter` attribute.

The class then implements the following functions:
\begin{align}
h_1(\varphi) &= \dfrac{\nu(\varphi)^2}{\epsilon(\varphi)}\\
h_2(\varphi) &= \dfrac{\nu(\varphi)^2}{2}\\
h_3(\varphi) &= \dfrac{\mu(\varphi)-h_1(\varphi)}{p(\varphi)^2}\\
\end{align}
where $\nu(\varphi),\epsilon(\varphi)$ and $\mu(\varphi)$ are defined in {cite:p}`bauchau2003vectorial`.

It then provides the expression for the corresponding rotation matrix $\boldsymbol{R}$:
\begin{equation}
\boldsymbol{R} = \boldsymbol{I} + h_1(\varphi)\boldsymbol{P} + h_2(\varphi)\boldsymbol{P}^2
\end{equation}
where $\boldsymbol{P} = \operatorname{skew}(\boldsymbol{p})$, as well as the associated rotation curvature matrix $\boldsymbol{H}$ involved in the computation of the rotation rate:
\begin{equation}
\boldsymbol{H} = \mu(\varphi)\boldsymbol{I} + h_2(\varphi)\boldsymbol{P} + h_3(\varphi)\boldsymbol{P}^2
\end{equation}

## Available particular cases

### `ExponentialMap` parametrization

This parametrization corresponds to the simple choice:
\begin{equation}
p(\varphi)=\varphi
\end{equation}
The corresponding expression for the rotation matrix is the famous Euler-Rodrigues formula.

### `EulerRodrigues` parametrization
This parametrization corresponds to the simple choice:
\begin{equation}
p(\varphi)=2 sin(\varphi/2)
\end{equation}

### `SineFamily` parametrization

This generic family for any integer $m$ corresponds to:
\begin{equation}
p(\varphi) = m \sin\left(\frac{\varphi}{m}\right)
\end{equation}

### `TangentFamily` parametrization

This generic family for any integer $m$ corresponds to:
\begin{equation}
p(\varphi) = m \tan\left(\frac{\varphi}{m}\right)
\end{equation}

## Code

```{raw-cell}
:raw_mimetype: text/restructuredtext

The corresponding module is available here :download:`rotation_parametrization.py`.

.. literalinclude:: rotation_parametrization.py
  :language: python
```

## References

```{bibliography}
:filter: docname in docnames
```