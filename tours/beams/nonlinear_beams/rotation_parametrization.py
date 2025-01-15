#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 14:28:31 2020

@author: bleyerj
"""
from ufl import sqrt, dot, sin, cos, tan, as_matrix, Identity
from dolfinx.fem import Constant

DOLFIN_EPS = 1e-12


def Skew(n):
    """Antisymmetric tensor associated with a vector n"""
    return as_matrix([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])


class RotationParametrization(object):
    """
    A generic class for handling a vectorial parametrization of rotation
    tensors.

        Bauchau, O. A., & Trainelli, L. (2003). The vectorial parameterization of
        rotation. Nonlinear dynamics, 32(1), 71-92. DOI: 10.1023/A:1024265401576
    """

    def __init__(self, p, dp):
        """
        Parameters
        ----------
        p : parametric function (must be such that p(x)/x -> 1 when x->0)

        dp : the derivative of p
        """
        self.p = p
        self.dp = dp

    def ntheta(self, theta):
        """Rotation vector norm"""
        return sqrt(dot(theta, theta) + DOLFIN_EPS)

    def h1(self, theta):
        x = self.ntheta(theta)
        return sin(x) / self.p(x)

    def h2(self, theta):
        x = self.ntheta(theta)
        return 2 * (sin(x / 2) / self.p(x)) ** 2

    def mu(self, theta):
        x = self.ntheta(theta)
        return 1 / self.dp(x)

    def h3(self, theta):
        x = self.ntheta(theta)
        return (self.mu(x) - self.h1(x)) / self.p(x) ** 2

    def rotation_parameter(self, theta):
        """Reparametrized rotation vector"""
        x = self.ntheta(theta)
        return self.p(x) * theta / x

    def rotation_matrix(self, theta):
        """Rotation matrix"""
        P = Skew(self.rotation_parameter(theta))
        return Identity(3) + self.h1(theta) * P + self.h2(theta) * P * P

    def curvature_matrix(self, theta):
        """Curvature matrix involved in the computation of the rotation rate"""
        P = Skew(self.rotation_parameter(theta))
        return (
            self.mu(theta) * Identity(3) + self.h2(theta) * P + self.h3(theta) * P * P
        )


class ExponentialMap(RotationParametrization):
    """Exponential map parametrization (p(x)=x)"""

    def __init__(self):
        RotationParametrization.__init__(self, lambda x: x, lambda x: 1)


class SineFamily(RotationParametrization):
    """Sine family parametrization (p(x)=m*sin(x/m))"""

    def __init__(self, m):
        RotationParametrization.__init__(
            self, lambda x: m * sin(x / m), lambda x: cos(x / m)
        )


class EulerRodrigues(SineFamily):
    """Euler-Rodrigues parametrization (p(x)=2*sin(x/2))"""

    def __init__(self):
        SineFamily.__init__(self, 2)


class TangentFamily(RotationParametrization):
    """Tangent family parametrization (p(x)=m*tan(x/m))"""

    def __init__(self, m):
        RotationParametrization.__init__(
            self, lambda x: m * tan(x / m), lambda x: 1 + tan(x / m) ** 2
        )


class RodriguesParametrization(RotationParametrization):
    def __init__(self):
        pass

    def rotation_parameter(self, theta):
        """Reparametrized rotation vector"""
        return theta

    def rotation_matrix(self, theta):
        P = Skew(self.rotation_parameter(theta))
        h = lambda x: 4 / (4 + x**2)
        return Identity(3) + h(theta) * P + h(theta) / 2 * P * P

    def cruvature_matrix(self, theta):
        """Curvature matrix involved in the computation of the rotation rate"""
        P = Skew(self.rotation_parameter(theta))
        h = lambda x: 4 / (4 + x**2)
        return h(theta) * Identity(3) + h(theta) / 2 * P
