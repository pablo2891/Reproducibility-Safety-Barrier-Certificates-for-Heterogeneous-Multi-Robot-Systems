from __future__ import annotations

import unittest

import numpy as np

from hetero_sbc.barriers import cbf_value, pairwise_safe_distance
from hetero_sbc.controllers import heterogeneous_barrier_controller
from hetero_sbc.scenarios import baseline_six, named_scenario
from hetero_sbc.simulator import simulate_scenario


class BarrierTests(unittest.TestCase):
    def test_cbf_positive_when_pair_is_safe_and_static(self) -> None:
        delta_p = np.array([1.0, 0.0])
        delta_v = np.array([0.0, 0.0])
        value = cbf_value(delta_p, delta_v, 1.2, 0.6, 0.4)
        self.assertGreater(value, 0.0)

    def test_safe_distance_uses_both_radii(self) -> None:
        radii = np.array([0.2, 0.4])
        self.assertAlmostEqual(pairwise_safe_distance(radii, 0, 1, 0.05), 0.65)

    def test_barrier_matches_nominal_when_agents_are_far_apart(self) -> None:
        cfg = baseline_six()
        positions = np.array([[-4.0, 0.0], [4.0, 0.0]], dtype=float)
        result = heterogeneous_barrier_controller(
            positions=positions,
            velocities=np.zeros((2, 2), dtype=float),
            goals=np.array([[-5.0, 0.0], [5.0, 0.0]], dtype=float),
            accel_limits=cfg.accel_limits[:2],
            radii=cfg.radii[:2],
            gamma=cfg.gamma[:2],
            safety_buffer=0.0,
            kp=cfg.kp,
            kd=cfg.kd,
        )
        self.assertLess(np.linalg.norm(result.control - result.nominal), 1e-2)

    def test_demo_scenario_completes_without_collision_under_barrier_controller(self) -> None:
        cfg = named_scenario("demo")
        result = simulate_scenario(cfg, "heterogeneous_barrier")
        self.assertFalse(result.summary["collision"])
        self.assertTrue(result.summary["all_goals_reached"])
        self.assertIsNotNone(result.summary["completion_step"])

    def test_scalability_scenario_completes_without_collision_under_barrier_controller(self) -> None:
        cfg = named_scenario("scalability_10")
        result = simulate_scenario(cfg, "heterogeneous_barrier")
        self.assertFalse(result.summary["collision"])
        self.assertTrue(result.summary["all_goals_reached"])
        self.assertIsNotNone(result.summary["completion_step"])

    def test_baseline_nominal_controller_terminates_on_collision(self) -> None:
        cfg = named_scenario("baseline")
        result = simulate_scenario(cfg, "nominal")
        self.assertTrue(result.summary["collision"])
        self.assertFalse(result.summary["all_goals_reached"])
        self.assertIsNone(result.summary["completion_step"])
        self.assertEqual(result.summary["termination_reason"], "collision")

    def test_baseline_heterogeneous_barrier_completes_swap(self) -> None:
        cfg = named_scenario("baseline")
        result = simulate_scenario(cfg, "heterogeneous_barrier")
        self.assertFalse(result.summary["collision"])
        self.assertTrue(result.summary["all_goals_reached"])
        self.assertIsNotNone(result.summary["completion_step"])
        self.assertEqual(result.summary["termination_reason"], "all_goals_reached")
        self.assertIn("min_pair_distance", result.summary)


if __name__ == "__main__":
    unittest.main()
