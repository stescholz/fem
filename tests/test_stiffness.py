import numpy as np
import mesh
from fem import *


def laplace(n):

    ident = np.eye(n)
    u_ones = np.diag(np.ones(n-1), 1)

    A = np.kron(u_ones, -ident)
    A += A.T+np.kron(ident, 4*ident - u_ones - u_ones.T)

    return A


def test_stiffness():

    m = mesh.RectangularMesh(0, 1, 0, 1, 0.1)
    fem = FEM(m)
    indizes = m.interior()
    # should yield the typical 5 point stencil
    stiffness = fem.stiffness().toarray()
    A = laplace(m.nx-2)

    assert np.allclose(stiffness[np.ix_(indizes,indizes)], A) == True
