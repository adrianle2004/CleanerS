"""Unit tests for inference/voxel_surface.py -- the occupancy-to-surface rules.

Pure numpy: no open3d, no window, no GPU, no checkpoint. Run with

    python -m unittest discover -s tests -v

What these pin down is the one property the new layer rests on: the surface
has to land on the SAME faces the cubes are drawn at, or pressing `m` moves
the room instead of re-describing it. display_solid draws a voxel's cube
spanning index +/- 0.5 (its _C corner table), so a flat wall's surface must
come out at exactly that half.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.voxel_surface import (nearest_colour, pull_to_surface, smooth,
                                    subdivide,
                                     surface_nets, voxelise)


def edge_counts(tris):
    """How many triangles each undirected edge belongs to."""
    counts = {}
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            counts[(min(u, v), max(u, v))] = \
                counts.get((min(u, v), max(u, v)), 0) + 1
    return counts


def normals(verts, tris):
    p = verts[tris]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


class TestEmpty(unittest.TestCase):
    def test_an_empty_grid_has_no_surface(self):
        verts, tris, owner = surface_nets(np.zeros((5, 5, 5), bool))
        self.assertEqual(len(verts), 0)
        self.assertEqual(len(tris), 0)
        self.assertEqual(len(owner), 0)

    def test_a_full_grid_is_still_surfaced_at_its_border(self):
        """Padding is not optional: without it a room whose floor runs to
        the edge of the map has no underside and you see into it."""
        occ = np.ones((4, 4, 4), bool)
        verts, tris, _ = surface_nets(occ)
        self.assertGreater(len(tris), 0)
        np.testing.assert_allclose(verts.min(axis=0), [-0.5, -0.5, -0.5])
        np.testing.assert_allclose(verts.max(axis=0), [3.5, 3.5, 3.5])


class TestSingleVoxel(unittest.TestCase):
    def occ(self):
        o = np.zeros((5, 5, 5), bool)
        o[2, 2, 2] = True
        return o

    def test_one_voxel_is_a_closed_shell(self):
        """Six crossing edges, so six quads and eight vertices -- and
        every edge shared by exactly two triangles, or the renderer
        shows a hole from one side."""
        verts, tris, _ = surface_nets(self.occ())
        self.assertEqual(len(verts), 8)
        self.assertEqual(len(tris), 12)
        self.assertEqual(set(edge_counts(tris).values()), {2})

    def test_one_voxel_is_centred_on_itself(self):
        verts, _, _ = surface_nets(self.occ())
        np.testing.assert_allclose(verts.mean(axis=0), [2.0, 2.0, 2.0])

    def test_its_normals_point_away_from_the_voxel(self):
        verts, tris, _ = surface_nets(self.occ())
        n = normals(verts, tris)
        centroid = verts[tris].mean(axis=1) - np.array([2.0, 2.0, 2.0])
        self.assertTrue((np.sum(n * centroid, axis=1) > 0).all())


class TestSlabAlignment(unittest.TestCase):
    """The property the whole layer rests on, as an assertion."""

    def occ(self):
        o = np.zeros((6, 6, 6), bool)
        o[:, :, :3] = True            # z = 0, 1, 2 solid
        return o

    def test_a_flat_face_lands_on_the_cube_face(self):
        """Voxel 2's cube spans 1.5..2.5, so the slab's top surface is at
        2.5 -- not 2.0, and not 3.0."""
        verts, _, _ = surface_nets(self.occ())
        self.assertAlmostEqual(verts[:, 2].max(), 2.5)
        self.assertAlmostEqual(verts[:, 2].min(), -0.5)

    def test_the_top_face_is_flat_and_faces_up(self):
        verts, tris, _ = surface_nets(self.occ())
        n = normals(verts, tris)
        top = np.all(np.isclose(verts[tris][:, :, 2], 2.5), axis=1)
        self.assertTrue(top.any())
        np.testing.assert_allclose(n[top], np.tile([0.0, 0.0, 1.0],
                                                   (top.sum(), 1)), atol=1e-9)


class TestOwner(unittest.TestCase):
    def test_every_vertex_names_an_occupied_voxel(self):
        """owner is what carries the colour: a vertex that named empty
        space would paint the surface with a voxel nobody predicted."""
        rng = np.random.RandomState(0)
        occ = rng.rand(7, 8, 9) > 0.6
        verts, _, owner = surface_nets(occ)
        self.assertEqual(len(owner), len(verts))
        self.assertTrue(occ[owner[:, 0], owner[:, 1], owner[:, 2]].all())

    def test_owner_indices_are_inside_the_grid(self):
        occ = np.zeros((4, 4, 4), bool)
        occ[0, 0, 0] = occ[3, 3, 3] = True
        _, _, owner = surface_nets(occ)
        self.assertTrue((owner >= 0).all())
        self.assertTrue((owner < np.array([4, 4, 4])).all())


class TestSmooth(unittest.TestCase):
    def slab(self):
        # wide on purpose: the slab's SIDE faces meet its top at the grid
        # border, and each smoothing pass pulls that join one ring further
        # in. A face only counts as flat where the border cannot reach it.
        o = np.zeros((16, 16, 6), bool)
        o[:, :, :3] = True
        return surface_nets(o)

    def test_smoothing_keeps_every_vertex(self):
        verts, tris, _ = self.slab()
        out = smooth(verts, tris, passes=3)
        self.assertEqual(out.shape, verts.shape)

    def test_smoothing_leaves_a_flat_face_flat(self):
        """A plane is its own neighbourhood average, so the face the
        cubes agree with must not drift off them."""
        verts, tris, _ = self.slab()
        out = smooth(verts, tris, passes=5)
        interior = np.isclose(verts[:, 2], 2.5) \
            & (np.abs(verts[:, 0] - 7.5) < 3.0) \
            & (np.abs(verts[:, 1] - 7.5) < 3.0)
        self.assertTrue(interior.any())
        np.testing.assert_allclose(out[interior, 2], 2.5, atol=1e-9)

    def test_smoothing_rounds_a_corner_off(self):
        """The point of it: an isolated voxel's eight corners move IN."""
        occ = np.zeros((5, 5, 5), bool)
        occ[2, 2, 2] = True
        verts, tris, _ = surface_nets(occ)
        out = smooth(verts, tris, passes=2)
        c = np.array([2.0, 2.0, 2.0])
        self.assertLess(np.abs(out - c).max(), np.abs(verts - c).max())

    def test_zero_passes_changes_nothing(self):
        verts, tris, _ = self.slab()
        np.testing.assert_array_equal(smooth(verts, tris, passes=0), verts)


class TestPullToSurface(unittest.TestCase):
    """Taking the model's thickness off the FRONT of a measured surface.

    The completion comes out about a voxel proud of what the depth
    measured, and the encoder already knows by how much: the TSDF is the
    signed distance to the nearest measured surface over VOX_MARGIN, so
    tsdf * 3 is that distance in 8 cm voxels. Positive is the camera's
    side (run_live.build_mask reads `tsdf < -0.5` as occluded), so the
    front is exactly the part with tsdf > 0 -- and the rest, which the
    sensor never saw, has nothing to be pulled towards and is left.
    """

    def wall(self, zero_at=10.0, shape=(20, 8, 8)):
        """A TSDF whose zero crossing is a plane at x = zero_at, one
        voxel of distance per 1/3 of TSDF."""
        i = np.arange(shape[0], dtype=np.float64)
        t = np.clip((i - zero_at) / 3.0, -1.0, 1.0)
        return np.broadcast_to(t[:, None, None], shape).copy()

    def test_a_vertex_in_front_is_pulled_onto_the_surface(self):
        t = self.wall()
        v = np.array([[11.0, 4.0, 4.0]])          # tsdf = +1/3, one voxel out
        out = pull_to_surface(v, t)
        self.assertAlmostEqual(out[0, 0], 10.0, places=5)

    def test_a_vertex_behind_the_surface_is_left_alone(self):
        """'the rest can be left there' -- behind it the sensor measured
        nothing, so there is no surface to shrink onto."""
        t = self.wall()
        v = np.array([[8.0, 4.0, 4.0]])           # tsdf = -2/3
        np.testing.assert_allclose(pull_to_surface(v, t), v)

    def test_a_vertex_already_on_the_surface_does_not_move(self):
        t = self.wall()
        v = np.array([[10.0, 4.0, 4.0]])
        np.testing.assert_allclose(pull_to_surface(v, t), v, atol=1e-9)

    def test_the_step_is_clamped(self):
        """A vertex 2.5 voxels out would be stepped 2.5, which is further
        than a completion is ever proud of a surface -- past that the
        TSDF is saturating and is no longer a distance anyone measured."""
        t = self.wall()
        v = np.array([[12.5, 4.0, 4.0]])          # tsdf = +5/6
        out = pull_to_surface(v, t, max_step=1.5)
        self.assertAlmostEqual(out[0, 0], 11.0, places=5)

    def test_a_saturated_vertex_is_left_alone(self):
        """Out in open space the TSDF is a flat 1: no gradient, so no
        direction, so nothing to move towards."""
        t = self.wall()
        v = np.array([[18.0, 4.0, 4.0]])
        np.testing.assert_allclose(pull_to_surface(v, t), v)

    def test_it_only_moves_along_the_gradient(self):
        t = self.wall()
        v = np.array([[11.0, 4.0, 4.0]])
        out = pull_to_surface(v, t)
        np.testing.assert_allclose(out[0, 1:], v[0, 1:], atol=1e-9)

    def test_the_input_is_not_modified(self):
        t = self.wall()
        v = np.array([[11.0, 4.0, 4.0]])
        before = v.copy()
        pull_to_surface(v, t)
        np.testing.assert_array_equal(v, before)

    def test_a_flat_region_with_no_gradient_is_left_alone(self):
        """Nothing measured anywhere: the TSDF is a constant, there is no
        direction to move in, and dividing by its length must not blow
        up."""
        t = np.ones((8, 8, 8))
        v = np.array([[4.0, 4.0, 4.0]])
        np.testing.assert_allclose(pull_to_surface(v, t), v)

    def test_no_vertices_is_not_an_error(self):
        out = pull_to_surface(np.zeros((0, 3)), self.wall())
        self.assertEqual(out.shape, (0, 3))


class TestVoxelise(unittest.TestCase):
    """Putting the sensor's own mesh onto the room's grid.

    The voxel layers and the sensor's mesh are built from the same depth
    frames and still disagree: the TSDF resolves 2 cm cells and raises a
    surface wherever a weighted average crosses zero, while the room
    samples ONE point per 8 cm voxel and needs two hits to outvote two
    passes. At a window that is the difference between a sheet and a
    hole. This is how the mesh's own verdict gets onto the grid, so the
    voxel layers can be made to cover what it covers.
    """

    def test_a_point_marks_the_voxel_it_falls_in(self):
        cov, _ = voxelise(np.array([[1.4, 2.6, 0.1]]), (4, 4, 4))
        self.assertTrue(cov[1, 2, 0])
        self.assertEqual(cov.sum(), 1)

    def test_points_outside_the_grid_are_dropped_not_wrapped(self):
        cov, _ = voxelise(np.array([[-1.0, 0.0, 0.0], [9.0, 0.0, 0.0]]),
                          (4, 4, 4))
        self.assertFalse(cov.any())

    def test_no_points_marks_nothing(self):
        cov, _ = voxelise(np.zeros((0, 3)), (3, 3, 3))
        self.assertFalse(cov.any())

    def test_values_are_averaged_per_voxel(self):
        """A mesh puts many vertices in one 8 cm voxel, and the colour
        that voxel should wear is their mean, not whichever came last."""
        pts = np.array([[1.1, 1.1, 1.1], [1.9, 1.9, 1.9]])
        vals = np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]])
        cov, mean = voxelise(pts, (4, 4, 4), values=vals)
        self.assertTrue(cov[1, 1, 1])
        np.testing.assert_allclose(mean[1, 1, 1], [5.0, 10.0, 15.0])

    def test_voxels_with_no_point_keep_a_zero_mean(self):
        pts = np.array([[1.5, 1.5, 1.5]])
        vals = np.array([[7.0, 7.0, 7.0]])
        _, mean = voxelise(pts, (4, 4, 4), values=vals)
        np.testing.assert_allclose(mean[0, 0, 0], [0.0, 0.0, 0.0])

    def test_without_values_there_is_no_mean(self):
        cov, mean = voxelise(np.array([[1.5, 1.5, 1.5]]), (4, 4, 4))
        self.assertTrue(cov[1, 1, 1])
        self.assertIsNone(mean)


class TestNearestColour(unittest.TestCase):
    """Giving the surface layer the colour the SENSOR'S mesh has there.

    The two layers are coloured from different places -- the room blends
    one colour per 8 cm voxel and falls back to a class tint where the
    camera never saw, the mesh carries the TSDF's own 2 cm texture -- so
    flipping between them changes the room's appearance as well as its
    geometry. Taking each surface vertex's colour from the nearest point
    on the mesh makes the two agree wherever the mesh has an answer.

    Wherever it does not -- the completion, which is the whole point of
    the layer -- the vertex must keep the colour it had. A nearest
    neighbour with no distance limit would drag a wall's colour across
    the room onto a completion standing in open space.
    """

    def ref(self):
        pts = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        cols = np.array([[255.0, 0.0, 0.0], [0.0, 255.0, 0.0]])
        return pts, cols

    def test_a_point_takes_its_nearest_colour(self):
        pts, cols = self.ref()
        out, found = nearest_colour(np.array([[0.2, 0.0, 0.0]]), pts, cols,
                                    max_dist=1.0,
                                    base=np.zeros((1, 3)))
        self.assertTrue(found[0])
        np.testing.assert_allclose(out[0], [255.0, 0.0, 0.0])

    def test_the_nearest_is_the_one_that_wins(self):
        pts, cols = self.ref()
        out, _ = nearest_colour(np.array([[9.8, 0.0, 0.0]]), pts, cols,
                                max_dist=1.0, base=np.zeros((1, 3)))
        np.testing.assert_allclose(out[0], [0.0, 255.0, 0.0])

    def test_beyond_the_limit_the_base_colour_is_kept(self):
        """A completion standing in open space has no measured colour
        anywhere near it, and must not be given a distant wall's."""
        pts, cols = self.ref()
        base = np.array([[7.0, 8.0, 9.0]])
        out, found = nearest_colour(np.array([[5.0, 0.0, 0.0]]), pts, cols,
                                    max_dist=1.0, base=base)
        self.assertFalse(found[0])
        np.testing.assert_allclose(out[0], base[0])

    def test_an_empty_reference_leaves_everything_alone(self):
        base = np.array([[1.0, 2.0, 3.0]])
        out, found = nearest_colour(np.array([[0.0, 0.0, 0.0]]),
                                    np.zeros((0, 3)), np.zeros((0, 3)),
                                    max_dist=1.0, base=base)
        self.assertFalse(found.any())
        np.testing.assert_allclose(out, base)

    def test_no_points_is_not_an_error(self):
        pts, cols = self.ref()
        out, found = nearest_colour(np.zeros((0, 3)), pts, cols,
                                    max_dist=1.0, base=np.zeros((0, 3)))
        self.assertEqual(out.shape, (0, 3))
        self.assertEqual(found.shape, (0,))

    def test_the_base_is_not_modified(self):
        pts, cols = self.ref()
        base = np.array([[1.0, 2.0, 3.0]])
        before = base.copy()
        nearest_colour(np.array([[0.1, 0.0, 0.0]]), pts, cols, 1.0, base)
        np.testing.assert_array_equal(base, before)


class TestSubdivide(unittest.TestCase):
    """One midpoint split, and a way to carry a per-vertex array over it.

    The surface has a vertex every 8 cm and the sensor's mesh carries 2
    cm of texture, so asking it for a colour once per cell throws most
    of it away -- the layer came out a blurred copy of the same room.
    Splitting first puts the vertex count where the mesh's is.

    `parents` is what makes it usable: every vertex names the two it
    sits between, an original naming itself twice, so a colour averages
    and a flag is carried by the same indexing.
    """

    def square(self):
        """Two triangles sharing an edge."""
        verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                          [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        tris = np.array([[0, 1, 2], [0, 2, 3]])
        return verts, tris

    def area(self, verts, tris):
        a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()

    def test_every_triangle_becomes_four(self):
        v, t = self.square()
        _, tris, _ = subdivide(v, t)
        self.assertEqual(len(tris), 4 * len(t))

    def test_a_shared_edge_is_split_once(self):
        """Twice would tear the surface along it."""
        v, t = self.square()
        verts, _, _ = subdivide(v, t)
        self.assertEqual(len(verts), 4 + 5)   # 4 corners, 5 edges

    def test_the_original_vertices_come_first_and_name_themselves(self):
        v, t = self.square()
        verts, _, parents = subdivide(v, t)
        np.testing.assert_allclose(verts[:len(v)], v)
        np.testing.assert_array_equal(parents[:len(v), 0],
                                      parents[:len(v), 1])
        np.testing.assert_array_equal(parents[:len(v), 0], np.arange(len(v)))

    def test_a_new_vertex_sits_between_the_two_it_names(self):
        v, t = self.square()
        verts, _, parents = subdivide(v, t)
        want = (v[parents[len(v):, 0]] + v[parents[len(v):, 1]]) * 0.5
        np.testing.assert_allclose(verts[len(v):], want)

    def test_it_is_the_same_surface(self):
        """Midpoints lie on the triangle, so the area cannot change."""
        v, t = self.square()
        verts, tris, _ = subdivide(v, t)
        self.assertAlmostEqual(self.area(verts, tris), self.area(v, t))

    def test_the_winding_survives(self):
        """Every piece faces the way its parent did, or the lighting
        turns the surface inside out."""
        v, t = self.square()
        verts, tris, _ = subdivide(v, t)
        a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
        n = np.cross(b - a, c - a)
        self.assertTrue((n[:, 2] > 0).all())

    def test_a_colour_carries_over_by_averaging(self):
        v, t = self.square()
        _, _, parents = subdivide(v, t)
        col = np.zeros((len(v), 3))
        col[1] = (255.0, 0.0, 0.0)
        out = col[parents].mean(axis=1)
        np.testing.assert_allclose(out[:len(v)], col)
        self.assertEqual(out[len(v):].max(), 127.5)

    def test_an_empty_mesh_is_not_an_error(self):
        verts, tris, parents = subdivide(np.zeros((0, 3)),
                                         np.zeros((0, 3), np.int64))
        self.assertEqual(len(verts), 0)
        self.assertEqual(len(tris), 0)
        self.assertEqual(parents.shape, (0, 2))


class TestNearestColourLimits(unittest.TestCase):
    """The tree and the limit, which the caller now has reason to vary.

    The room is recoloured every few frames against the same 62 k-vertex
    mesh, so it builds the tree once and hands it back. And the limit is
    not one number any more: a voxel the camera coloured keeps the short
    reach, a voxel no photograph ever reached has none.
    """

    def ref(self):
        pts = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        cols = np.array([[255.0, 0.0, 0.0], [0.0, 255.0, 0.0]])
        return pts, cols

    def test_a_prebuilt_tree_answers_the_same(self):
        from scipy.spatial import cKDTree
        pts, cols = self.ref()
        q = np.array([[0.2, 0.0, 0.0], [5.0, 0.0, 0.0]])
        base = np.zeros((2, 3))
        want = nearest_colour(q, pts, cols, 1.0, base)
        got = nearest_colour(q, pts, cols, 1.0, base, tree=cKDTree(pts))
        np.testing.assert_allclose(got[0], want[0])
        np.testing.assert_array_equal(got[1], want[1])

    def test_an_all_finite_limit_still_finds_what_is_in_range(self):
        """The query is capped at the LARGEST limit anyone asked for,
        which is what lets a frame -- where nothing reaches further
        than the reach -- skip an unbounded search of the whole mesh."""
        pts, cols = self.ref()
        q = np.array([[0.2, 0.0, 0.0], [5.0, 0.0, 0.0]])
        base = np.zeros((2, 3))
        out, found = nearest_colour(q, pts, cols,
                                    np.array([1.0, 1.0]), base)
        np.testing.assert_array_equal(found, [True, False])
        np.testing.assert_allclose(out[0], [255.0, 0.0, 0.0])

    def test_the_limit_may_be_one_per_point(self):
        pts, cols = self.ref()
        q = np.array([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        base = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        out, found = nearest_colour(q, pts, cols,
                                    np.array([1.0, np.inf]), base)
        np.testing.assert_array_equal(found, [False, True])
        np.testing.assert_allclose(out[0], base[0])
        np.testing.assert_allclose(out[1], [255.0, 0.0, 0.0])


if __name__ == '__main__':
    unittest.main()
