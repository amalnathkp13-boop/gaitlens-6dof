"""Kinematics helpers: Grood-Suntay / ISB joint coordinate system for the knee.

Segment frames (right-handed, standing): x = anterior, y = superior (long axis), z = lateral.
Knee rotation  R_knee = Rz(-flex) . Rx(add) . Ry(ie)   (intrinsic Z-X'-Y'', ISB JCS)
  flex > 0 : flexion        add > 0 : adduction / varus        ie > 0 : tibial internal rotation
Translations (mm) of tibial origin relative to femoral origin, on JCS axes:
  q1 = medio-lateral (femoral e1),  q2 = antero-posterior (floating e2),  q3 = proximo-distal (tibial e3)
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

UP = np.array([0.0, 1.0, 0.0])
G = 9.81


def knee_rot(flex, add, ie):
    return R.from_euler("ZXY", np.column_stack([-np.asarray(flex), add, ie]), degrees=True)


def knee_angles(Rk):
    e = Rk.as_euler("ZXY", degrees=True)
    return -e[:, 0], e[:, 1], e[:, 2]


def _axes(Rk):
    M = Rk.as_matrix()
    n = len(M)
    e1 = np.tile([0.0, 0.0, 1.0], (n, 1))
    e3 = M[:, :, 1]
    e2 = np.cross(e3, e1)
    e2 /= np.linalg.norm(e2, axis=1, keepdims=True)
    return e1, e2, e3


def trans_to_vec(Rk, q_mm):
    e1, e2, e3 = _axes(Rk)
    return (q_mm[:, 0:1] * e1 + q_mm[:, 1:2] * e2 + q_mm[:, 2:3] * e3) / 1000.0


def vec_to_trans(Rk, d_m):
    e1, e2, e3 = _axes(Rk)
    A = np.stack([e1, e2, e3], axis=2)
    return np.linalg.solve(A, d_m[..., None])[..., 0] * 1000.0


def Ry(a):
    a = np.atleast_1d(np.asarray(a, float))
    z = np.zeros_like(a)
    return R.from_rotvec(np.column_stack([z, a, z]))


def yaw_of(v):
    return np.arctan2(-v[..., 2], v[..., 0])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)
