from __future__ import division, print_function

import numpy as np
from scipy import sparse
import time

from mesh import *


class FEM:
    """Linear finite elements in two dimensions

    A simply method to solve partial differential equations in two dimensions
    with linear finite elements on triangles.

    Parameters
    ----------
    mesh: Mesh, object
        The underlying mesh of the domain.

    Attributes
    ----------
    mesh: Mesh(object)
        The underlying mesh of the domain.
    """
    def __init__(self, mesh):
        self.mesh = mesh

    def element_stiffness(self, vertices):
        """Compute the stiffness matrix for an element in the mesh

        Computes the sum of the integral over the element T of u_x^2 + u_y^2
        over all possible basis functions (node functions) on T.

        Parameters
        ----------
        vertices: array like
            The coordinates of the three vertices counterclockwise.

        Returns
        -------
        array, shape(3, 3)
            The element stiffness matrix.
        """
        v13 = vertices[0]-vertices[2]
        v23 = vertices[1]-vertices[2]
        g1 = np.dot(v23, v23)
        g2 = -np.dot(v13, v23)
        g3 = np.dot(v13, v13)
        return (g1*np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) +
                g2*np.array([[0, 1, -1], [1, 0, -1], [-1, -1, 2]]) +
                g3*np.array([[0, 0, 0], [0, 1, -1], [0, -1, 1]])
                )/(2*(v13[0]*v23[1]-v23[0]*v13[1]))

    def element_mass(self, vertices):
        pass

    def element_load(self, vertices, f):
        pass

    def stiffness(self):
        """Build the stiffness matrix for the problem

        Returns
        -------
        array
            The stiffness matrix.
        """
        n = len(self.mesh.nodes)
        stm = sparse.lil_matrix((n, n))
        for el in self.mesh.elements:
            stm[np.ix_(el, el)] += self.element_stiffness(self.mesh.nodes[el])
        return stm


if __name__ == '__main__':
    m = RectangularMesh(0, 2, 0, 2, 0.06)
    fem = FEM(m)
    stiffness = fem.stiffness()
    indizes = m.interior()
    print(stiffness[np.ix_(indizes, indizes)].toarray())
    m.plot()
