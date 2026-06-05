import unittest

import torch

from geometry.manifold_utils import (
    complex_normalize,
    conformal_scale,
    cp_distance,
    distance,
    euc_exp_map,
    euc_log_map,
    gather_curvature,
    grassmann_distance,
    grassmann_project,
    poincare_distance,
    poincare_exp_map,
    poincare_log_map,
    poincare_proj,
    sphere_distance,
    sphere_exp_map,
    sphere_log_map,
    sphere_project,
    torus_distance,
    torus_exp_map,
    torus_log_map,
    warp_distances,
    wrap_angles,
)

class TestManifoldUtils(unittest.TestCase):
    def test_conformal_bounds(self):
        phi = torch.randn(5)
        for b in (0.05, 0.1, 0.2):
            s = conformal_scale(phi, b=b)
            self.assertTrue(torch.all(s >= (1 - b) - 1e-6))
            self.assertTrue(torch.all(s <= (1 + b) + 1e-6))

    def test_euclidean_logexp_inverse(self):
        x = torch.randn(8, 16)
        y = torch.randn(8, 16)
        v = euc_log_map(x, y)
        y2 = euc_exp_map(x, v)
        self.assertTrue(torch.allclose(y, y2, atol=1e-6))

    def test_hyperbolic_logexp_inverse(self):
        c = 0.1
        x = poincare_proj(torch.randn(8, 16) * 0.2, c)
        y = poincare_proj(torch.randn(8, 16) * 0.2, c)
        v = poincare_log_map(x, y, c)
        y2 = poincare_exp_map(x, v, c)
        d1 = poincare_distance(x, y, c)
        d2 = poincare_distance(x, y2, c)
        self.assertTrue(torch.allclose(d1, d2, atol=1e-4))

    def test_sphere_logexp_inverse(self):
        x = sphere_project(torch.randn(8, 16))
        y = sphere_project(torch.randn(8, 16))
        v = sphere_log_map(x, y)
        y2 = sphere_exp_map(x, v)
        d1 = sphere_distance(x, y)
        d2 = sphere_distance(x, y2)
        self.assertTrue(torch.allclose(d1, d2, atol=1e-4))

    def test_torus_roundtrip(self):
        thx = wrap_angles(torch.randn(8, 10))
        thy = wrap_angles(torch.randn(8, 10))
        v = torus_log_map(thx, thy)
        thy2 = torus_exp_map(thx, v)
        d1 = torus_distance(thx, thy)
        d2 = torus_distance(thx, thy2)
        self.assertTrue(torch.allclose(d1, d2, atol=1e-6))

    def test_cp_phase_invariance(self):
        z = complex_normalize(torch.randn(16, 2))
        w = complex_normalize(torch.randn(16, 2))
        d1 = cp_distance(z, w)
        phi = torch.rand(1) * 2 * torch.pi
        rot = torch.tensor([torch.cos(phi), torch.sin(phi)])
        r = torch.stack([rot[0], -rot[1], rot[1], rot[0]]).reshape(2, 2)
        z_rot = z @ r
        d2 = cp_distance(z_rot, w)
        self.assertTrue(torch.allclose(d1, d2, atol=1e-6))

    def test_grassmann_distance_nonnegative(self):
        n, k = 32, 4
        u = grassmann_project(torch.randn(n, k))
        v = grassmann_project(torch.randn(n, k))
        d = grassmann_distance(u, v)
        self.assertTrue((d.ndim == 1 and d.shape[0] == 1) or (d.ndim == 2 and d.shape[-1] == 1))
        self.assertTrue(torch.all(d >= 0))

    def test_warp_broadcast(self):
        bsz, k, m = 3, 5, 50
        base = torch.rand(bsz, k)
        idx = torch.randint(0, m, (bsz, k))
        curv = torch.randn(m)
        csel = gather_curvature(curv, idx)
        conf = conformal_scale(torch.randn(bsz, k), b=0.1)
        w = warp_distances(base, csel, conf)
        self.assertEqual(w.shape, base.shape)
        self.assertTrue(torch.isfinite(w).all())

    def test_unified_dispatch_shapes(self):
        x = torch.randn(4, 8)
        y = torch.randn(4, 8)
        self.assertEqual(distance("euclid", x, y).shape, (4, 1))
        self.assertEqual(distance("sphere", x, y).shape, (4, 1))
        self.assertEqual(distance("hyperbolic", x * 0.1, y * 0.1, kappa=-0.1).shape, (4, 1))
        thx, thy = wrap_angles(x), wrap_angles(y)
        self.assertEqual(distance("torus", thx, thy).shape, (4, 1))


if __name__ == "__main__":
    unittest.main()

