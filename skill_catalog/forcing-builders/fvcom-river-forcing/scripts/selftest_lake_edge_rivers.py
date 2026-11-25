import tempfile
import unittest
from pathlib import Path
import numpy as np
from build_lake_edge_rivers import edge_faces, read_control


class SignedRiver(unittest.TestCase):
    def test_native_edges_and_rejection(self):
        edges = [{"cell_id":1}]
        tri = np.array([[3, 1, 2]])
        edge_faces(edges, tri, np.array([[2, 0, 3]]), np.ones(3))
        self.assertEqual(edges[0]["edge_nodes"], [3,2])
        with self.assertRaisesRegex(ValueError, "shoreline face"):
            edge_faces(edges, tri, np.array([[0,0,3]]), np.ones(3))

    def test_preserve_published_signed_factors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.river.ctl"
            path.write_text('2 1 1.0 !! NIJ NRIVERS DELT\n'
                            ' 1 01234 TEST USGS 0 10 5 0 28 12 1 1 "Sample"\n'
                            ' 1 4 4 0 3 1 -0.12 1 1.0 "Sample"\n'
                            ' 2 5 5 0 3 1 -0.13 1 1.0 "Sample"\n')
            stations, edges = read_control(path)
            self.assertEqual(stations[1]["id"], "01234")
            self.assertEqual([edge["flow_scale"] for edge in edges], [-.12,-.13])


if __name__ == "__main__":
    unittest.main()
