"""docstring"""
from __future__ import print_function
import numpy as np


def check_tri_winding(tri, allow_reversed):
    """docstring"""
    trisq = np.ones((3, 3))
    trisq[:, 0:2] = np.array(tri)
    det_tri = np.linalg.det(trisq)
    if det_tri < 0.0:
        if allow_reversed:
            a = trisq[2, :].copy()
            trisq[2, :] = trisq[1, :]
            trisq[1, :] = a
        else:
            raise ValueError("triangle has wrong winding direction")
    return trisq


def tri_tri_2d(t_1, t_2, eps=0.0, allow_reversed=False, on_boundary=True):
    """docstring"""
    # Trangles must be expressed anti-clockwise
    t_1s = check_tri_winding(t_1, allow_reversed)
    t_2s = check_tri_winding(t_2, allow_reversed)

    if on_boundary:
        # Points on the boundary are considered as colliding
        chk_edge = lambda x: np.linalg.det(x) < eps
    else:
        # Points on the boundary are not considered as colliding
        chk_edge = lambda x: np.linalg.det(x) <= eps

    # For edge E of trangle 1,
    for i in range(3):
        edge = np.roll(t_1s, i, axis=0)[:2, :]

        # Check all points of trangle 2 lay on the external side of the edge E. If
        # they do, the triangles do not collide.
        if (chk_edge(np.vstack((edge, t_2s[0]))) and
                chk_edge(np.vstack((edge, t_2s[1]))) and
                chk_edge(np.vstack((edge, t_2s[2])))):
            return False

    # For edge E of trangle 2,
    for i in range(3):
        edge = np.roll(t_2s, i, axis=0)[:2, :]

        # Check all points of trangle 1 lay on the external side of the edge E. If
        # they do, the triangles do not collide.
        if (chk_edge(np.vstack((edge, t_1s[0]))) and
                chk_edge(np.vstack((edge, t_1s[1]))) and
                chk_edge(np.vstack((edge, t_1s[2])))):
            return False

    # The triangles collide
    return True


if __name__ == "__main__":
    t1 = [[0, 0], [5, 0], [0, 5]]
    t2 = [[0, 0], [5, 0], [0, 6]]
    print(tri_tri_2d(t1, t2), True)

    t1 = [[0, 0], [0, 5], [5, 0]]
    t2 = [[0, 0], [0, 6], [5, 0]]
    print(tri_tri_2d(t1, t2, allow_reversed=True), True)

    t1 = [[0, 0], [5, 0], [0, 5]]
    t2 = [[-10, 0], [-5, 0], [-1, 6]]
    print(tri_tri_2d(t1, t2), False)

    t1 = [[0, 0], [5, 0], [2.5, 5]]
    t2 = [[0, 4], [2.5, -1], [5, 4]]
    print(tri_tri_2d(t1, t2), True)

    t1 = [[0, 0], [1, 1], [0, 2]]
    t2 = [[2, 1], [3, 0], [3, 2]]
    print(tri_tri_2d(t1, t2), False)

    t1 = [[0, 0], [1, 1], [0, 2]]
    t2 = [[2, 1], [3, -2], [3, 4]]
    print(tri_tri_2d(t1, t2), False)

    # Barely touching
    t1 = [[0, 0], [1, 0], [0, 1]]
    t2 = [[1, 0], [2, 0], [1, 1]]
    print(tri_tri_2d(t1, t2, on_boundary=True), True)

    # Barely touching
    t1 = [[0, 0], [1, 0], [0, 1]]
    t2 = [[1, 0], [2, 0], [1, 1]]
    print(tri_tri_2d(t1, t2, on_boundary=False), False)
