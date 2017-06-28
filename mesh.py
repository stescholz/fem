from __future__ import division, print_function

import numpy as np
import math


class Mesh:
    """Triangular mesh of a two dimensional domain

    Attributes
    ----------
    nodes: array like, shape(num_nodes, 2)
        List coordinates of all nodes in the mesh
    elements: array like, shape(num_elem, 3)
        List of elements. Every entry holds the indices of the vertices in
        counterclockwise order.
    boundary: array like, shape(num_b_elem, 2)
        List of boundary elements. Every entry holds the indices of the
        start and the end vertice.
    """
    def __init__(self):
        self.nodes = None
        self.elements = None
        self.boundary = None

    def interior(self):
        """Get all vertices in the interior of the domain

        Returns
        -------
        list
            A list of the indices of the vertices in the interior.
        """
        return list(set(range(len(self.nodes))) - set(self.boundary[:, 0]))

    def plot(self):
        """plot the mesh"""
        import matplotlib.pyplot as plt
        import matplotlib.tri as tri

        plt.gca().set_aspect('equal')
        plt.triplot(self.nodes[:, 0], self.nodes[:, 1], self.elements)
        plt.show()

    def max_diameter(self):
        """calculate the maximal diameter of the elements in the mesh

        Returns
        -------
        float
            The maximal diameter of the elements in the mesh.
        """
        h_m = 0

        for el in self.elements:
            a = self.nodes[el[0]]
            b = self.nodes[el[1]]
            c = self.nodes[el[2]]

            h = math.sqrt(np.dot(a-b, a-b)*np.dot(a-c, a-c)*np.dot(b-c, b-c))
            h /= (a[0]-c[0])*(b[1]-c[1])-(b[0]-c[0])*(a[1]-c[1])

            h_m = max(h_m, h)

        return h_m

    def read_from_file(self, filename):
        pass

    def read_triangle_files(self, filename):
        """Create a mesh from the output of Triangle.

        With Triangle you can create triangular meshes in two dimensions.
        The method reads the output of the program and fills the attributes.

        Parameters
        ----------
        filename: str
            The prefix of the file names which Traingle creates. For example
            quadrat.poly as input becomes quadrat.1.node and so on. So
            quadrat.1 is the filename. Don't forget the -e switch otherwise
            Triangle won't create the file for the boundary.
        """
        with open(filename+".node", "r") as f:
            n = int(f.readline().split()[0])
            self.nodes = np.zeros((n, 2))
            for i in range(n):
                s = f.readline().split()
                self.nodes[i] = (float(s[1]), float(s[2]))

        with open(filename+".ele", "r") as f:
            n = int(f.readline().split()[0])
            self.elements = np.zeros((n, 3), dtype=np.int)
            for i in range(n):
                s = f.readline().split()
                self.elements[i] = (int(s[1])-1, int(s[2])-1, int(s[3])-1)

        with open(filename+".edge", "r") as f:
            n = int(f.readline().split()[0])
            self.boundary = np.zeros((n, 2), dtype=np.int)
            for i in range(n):
                s = f.readline().split()
                self.boundary[i] = (int(s[1])-1, int(s[2])-1)


class RectangularMesh(Mesh):
    """Simple mesh on a rectangular grid.

    A mesh on a uniform grid of the square [x_min, x_max] x [y_min, y_max].

    Parameters
    ----------
    x_min: float
        Left boundary of the first dimension.
    x_max: float
        Right boundary of the first dimension.
    y_min: float
        Left boundary of the second dimension.
    y_max: float
        Right boundary of the second dimension.
    h_max: float
        Maximal diameter of the triangles.

    Attributes
    ----------
    x_min: float
        Left boundary of the first dimension.
    x_max: float
        Right boundary of the first dimension.
    y_min: float
        Left boundary of the second dimension.
    y_max: float
        Right boundary of the second dimension.
    nx: int
        Number of grid points in x direction.
    ny: int
        Number of grid points in y direction.
    """
    def __init__(self, x_min, x_max, y_min, y_max, h_max):

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        # compute steps for each direction
        self.nx = int(math.ceil(math.sqrt(2)*(x_max-x_min)/h_max))+1
        self.ny = int(math.ceil((y_max-y_min)*(self.nx-1)/(x_max-x_min)))+1

        self.compute_nodes()
        self.compute_elements()
        self.compute_boundary()

    def compute_nodes(self):

        x = np.linspace(self.x_min, self.x_max, self.nx)
        y = np.linspace(self.y_min, self.y_max, self.ny)
        xx, yy = np.meshgrid(x, y)
        self.nodes = np.array([xx.flatten(), yy.flatten()]).T

        # slow variant
        # self.nodes =
        #          np.array([[x,y] for y in np.linspace(0, self.b, self.ny)
        #                         for x in np.linspace(0, self.a, self.nx)])

    def compute_elements(self):

        self.elements = np.zeros((2*(self.nx-1)*(self.ny-1), 3), dtype=np.int)

        i = 0
        for l in range(self.ny-1):
            for k in range(self.nx-1):
                bottom_left = k+l*self.nx
                self.elements[i] = (bottom_left,
                                    bottom_left+self.nx+1,
                                    bottom_left+self.nx)
                self.elements[i+1] = (bottom_left,
                                      bottom_left+1,
                                      bottom_left+self.nx+1)
                i += 2

    def compute_boundary(self):

        bot = np.array(range(self.nx))
        top = bot + +self.nx*(self.ny-1)
        left = np.array(range(0, self.nx*self.ny, self.nx))
        right = left + (self.nx-1)

        start = np.hstack([bot[:-1], right[:-1], top[:0:-1], left[-1:0:-1]])

        self.boundary = np.array([start, np.roll(start, -1)]).T


if __name__ == '__main__':

    m = Mesh()
    m.read_triangle_files('meshes/lshaped.1')
    m.plot()
    print(m.max_diameter())
