import math
import unittest

from plot_colocated_campaign import CONDITIONS, MINI_APP, collect, mean_ci


class FigureDataTests(unittest.TestCase):
    def test_confidence_interval_uses_jobs(self):
        mean, half = mean_ci([1, 2, 3, 4, 5])
        self.assertEqual(mean, 3)
        self.assertAlmostEqual(half, 2.7764451051977987 * math.sqrt(2.5 / 5))

    def test_identical_values(self):
        self.assertEqual(mean_ci([7] * 5), (7, 0))

    def test_reject_invalid_samples(self):
        for values in ([1] * 4, [1] * 6, [1, 2, 3, 4, float("nan")]):
            with self.assertRaises(ValueError):
                mean_ci(values)

    def test_production_campaign(self):
        data = collect(MINI_APP / "logs/colocated_campaign/colocated_full_4961357")
        self.assertEqual(len(data["absolute_time"]), 10)
        self.assertEqual(len(data["job_means"]), 50)
        self.assertEqual(len(data["payload"]), 20)
        for platform in ("cpu", "gpu"):
            rows = [r for r in data["absolute_time"] if r["platform"] == platform]
            self.assertEqual({r["condition"] for r in rows}, set(CONDITIONS))
            self.assertTrue(all(r["ci95_low_s"] <= r["mean_s"] <= r["ci95_high_s"] for r in rows))
            payload = {r["condition"]: r for r in data["payload"] if r["platform"] == platform}
            without, with_preload = payload[CONDITIONS[0]], payload[CONDITIONS[1]]
            self.assertAlmostEqual(without["input_mib"] / with_preload["input_mib"], 2, places=4)
            self.assertEqual(without["output_mib"], with_preload["output_mib"])
            self.assertEqual(without["preload_mib"], 0)
            self.assertGreater(with_preload["preload_mib"], 0)


if __name__ == "__main__":
    unittest.main()
