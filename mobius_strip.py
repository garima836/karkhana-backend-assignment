
"""
Mobius Strip Modeling Script

This script defines a MobiusStrip class that:
- Generates a 3D mesh from parametric equations
- Computes surface area via numerical integration
- Computes edge length by summing distances
- Visualizes the Mobius strip using Matplotlib

Author: OpenAI's ChatGPT
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MobiusStrip:
    def __init__(self, R=1.0, w=0.5, n=100):
        self.R = R
        self.w = w
        self.n = n
        self.u, self.v = np.meshgrid(
            np.linspace(0, 2 * np.pi, n),
            np.linspace(-w / 2, w / 2, n)
        )
        self.x, self.y, self.z = self._compute_coordinates()

    def _compute_coordinates(self):
        u = self.u
        v = self.v
        R = self.R

        x = (R + v * np.cos(u / 2)) * np.cos(u)
        y = (R + v * np.cos(u / 2)) * np.sin(u)
        z = v * np.sin(u / 2)

        return x, y, z

    def surface_area(self):
        du = 2 * np.pi / (self.n - 1)
        dv = self.w / (self.n - 1)

        xu = np.gradient(self.x, axis=1)
        xv = np.gradient(self.x, axis=0)
        yu = np.gradient(self.y, axis=1)
        yv = np.gradient(self.y, axis=0)
        zu = np.gradient(self.z, axis=1)
        zv = np.gradient(self.z, axis=0)

        cross_x = yu * zv - zu * yv
        cross_y = zu * xv - xu * zv
        cross_z = xu * yv - yu * xv

        dA = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)
        surface_area = np.sum(dA) * du * dv
        return surface_area

    def edge_length(self):
        u = np.linspace(0, 2 * np.pi, self.n)
        v = self.w / 2

        x = (self.R + v * np.cos(u / 2)) * np.cos(u)
        y = (self.R + v * np.cos(u / 2)) * np.sin(u)
        z = v * np.sin(u / 2)

        dx = np.diff(x)
        dy = np.diff(y)
        dz = np.diff(z)

        ds = np.sqrt(dx**2 + dy**2 + dz**2)
        return 2 * np.sum(ds)

    def plot(self):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.x, self.y, self.z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_title("Möbius Strip")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    mobius = MobiusStrip(R=1.0, w=0.5, n=200)
    print("Surface Area:", mobius.surface_area())
    print("Edge Length:", mobius.edge_length())
    mobius.plot()
