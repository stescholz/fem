import numpy as np
import mesh


def test_rect_mesh():

    rect_mesh = mesh.RectangularMesh(0, 2, 0, 2, 2**0.5)

    # should yield a grid with mesh with 1 and 3 points in each direction
    nodes = np.array([[x, y] for y in range(3) for x in range(3)])
    assert np.allclose(rect_mesh.nodes, nodes) == True

    elements = np.array([[0, 4, 3], [0, 1, 4], [1, 5, 4],
                         [1, 2, 5], [3, 7, 6], [3, 4, 7],
                         [4, 8, 7], [4, 5, 8]])
    assert np.allclose(rect_mesh.elements, elements) == True
