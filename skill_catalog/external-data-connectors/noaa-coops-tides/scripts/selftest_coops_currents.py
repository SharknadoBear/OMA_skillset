from __future__ import annotations

import json

import numpy as np
import pandas as pd

from coops_currents import depth_average, seven_day_chunks, speed_direction_to_uv


def main() -> int:
    chunks = seven_day_chunks("2025-04-01T00:00:00Z", "2025-05-01T00:00:00Z")
    assert len(chunks) == 5
    assert all((b - a) <= pd.Timedelta(days=7) for a, b in chunks)
    u, v = speed_direction_to_uv(np.array([100.0, 100.0, 100.0, 100.0]), np.array([0.0, 90.0, 180.0, 270.0]))
    assert np.allclose(u, [0, 1, 0, -1], atol=1e-12)
    assert np.allclose(v, [1, 0, -1, 0], atol=1e-12)
    group = pd.DataFrame({"bin": [1, 2, 3], "east_m_s": [1.0, 2.0, 3.0], "north_m_s": [0.0, 0.0, 0.0]})
    avg = depth_average(group, {1: 2.0, 2: 5.0, 3: 8.0}, 10.0)
    # Interfaces 0, 3.5, 6.5, 10 -> weights 3.5, 3, 3.5.
    assert abs(avg["east_m_s"] - 2.0) < 1e-12 and avg["weight_sum_m"] == 10.0
    print(json.dumps({"status": "pass", "chunks": len(chunks), "weighted_east": avg["east_m_s"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
