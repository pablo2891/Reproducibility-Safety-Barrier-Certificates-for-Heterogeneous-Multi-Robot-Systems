from __future__ import annotations

import unittest

import numpy as np

from hetero_sbc.barriers import cbf_value, pairwise_safe_distance
from hetero_sbc.controllers import heterogeneous_barrier_controller
from hetero_sbc.scenarios import baseline_six, named_scenario, paper_baseline_six, paper_scalability_case
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

    def test_decentralized_barrier_demo_remains_collision_free(self) -> None:
        cfg = named_scenario("demo")
        result = simulate_scenario(cfg, "decentralized_heterogeneous_barrier")
        self.assertFalse(result.summary["collision"])
        self.assertTrue(result.summary["all_goals_reached"])
        self.assertEqual(result.summary["termination_reason"], "all_goals_reached")
        self.assertIn("mean_active_neighbors", result.summary)
        self.assertIn("qp_fallback_count", result.summary)

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

    def test_paper_baseline_decentralized_barrier_completes_swap(self) -> None:
        cfg = paper_baseline_six()
        result = simulate_scenario(cfg, "decentralized_heterogeneous_barrier")
        self.assertFalse(result.summary["collision"])
        self.assertTrue(result.summary["all_goals_reached"])
        self.assertEqual(result.summary["termination_reason"], "all_goals_reached")
        self.assertFalse(result.summary["collision_after_clamping"])

    def test_lane_swap_stress_test_records_first_infeasible_event(self) -> None:
        cfg = baseline_six()
        result = simulate_scenario(cfg, "decentralized_heterogeneous_barrier")
        self.assertTrue(result.summary["collision"])
        self.assertGreater(result.summary["qp_infeasible_count"], 0)
        self.assertIn("first_infeasible_step", result.summary)
        self.assertIn("first_infeasible_pair", result.summary)
        self.assertIn("first_infeasible_h_ij", result.summary)
        self.assertIn("infeasibility_before_collision", result.summary)
        self.assertIsNotNone(result.metadata["first_infeasible_event"])
        self.assertIsNotNone(result.metadata["collision_event"])
        self.assertIn("pair_h_ij_window", result.metadata["first_infeasible_event"])

    def test_paper_scalability_smoke_runs_record_summary_fields(self) -> None:
        for n_agents in (10, 15):
            cfg = paper_scalability_case(n_agents)
            result = simulate_scenario(cfg, "decentralized_heterogeneous_barrier")
            self.assertEqual(result.name, f"paper_scalability_{n_agents}")
            self.assertIn("collision", result.summary)
            self.assertIn("all_goals_reached", result.summary)
            self.assertIn("completion_step", result.summary)
            self.assertIn("min_clearance", result.summary)
            self.assertIn("min_pair_distance", result.summary)
            self.assertIn("qp_infeasible_count", result.summary)
            self.assertIn("deadlock_count", result.summary)
            self.assertIn("stopped_not_at_goal", result.metadata)
            self.assertIn("pairwise_min_cbf", result.metadata)


if __name__ == "__main__":
    unittest.main()
