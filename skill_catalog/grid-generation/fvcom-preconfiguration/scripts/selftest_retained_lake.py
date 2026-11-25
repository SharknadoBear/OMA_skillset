import unittest
import numpy as np
from prepare_retained_lake import topology


class NativeTopology(unittest.TestCase):
    def test_closed_lake_and_corrupt_native_neighbors(self):
        tri = np.array([[1, 3, 2], [1, 4, 3]])
        neighbors = np.array([[0, 0, 2], [0, 1, 0]])
        x = np.array([0., 1., 1., 0.])
        y = np.array([0., 0., 1., 1.])
        loops, result = topology(tri, neighbors, x, y)
        self.assertEqual(result["shoreline_edges"], 4)
        self.assertEqual(result["islands"], 0)
        self.assertEqual(set(loops[0]), {1, 2, 3, 4})
        with self.assertRaisesRegex(ValueError, "neighbor"):
            topology(tri, np.zeros_like(neighbors), x, y)
        with self.assertRaisesRegex(ValueError, "clockwise"):
            topology(tri[:, ::-1], neighbors[:, ::-1], x, y)


if __name__ == "__main__":
    unittest.main()
