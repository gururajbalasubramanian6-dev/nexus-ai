"""
Unit tests for Member 4: Financial Impact & Recovery Optimization Engine
(agents/financial_optimization_agent.py)

Covers all 14 mandatory test cases specified in the project requirements:
1. Correct calculation of gross disruption loss.
2. Correct calculation of recovery plan total cost.
3. Correct calculation of estimated savings.
4. Recovery option within the budget.
5. Recovery option exceeding the budget.
6. Recovery option exceeding available capacity (shortfall).
7. Recovery option violating a specified lead-time limit.
8. Selection of the lowest-total-cost feasible option.
9. No feasible option available.
10. Missing financial data is reported.
11. Invalid negative quantities or costs are handled appropriately.
12. No double-counting of financial losses.
13. No-action baseline comparison.
14. Correct structured output schema.
"""

import unittest
from typing import Any, Dict, List

from agents.financial_optimization_agent import (
    DisruptionScenario,
    FinancialValidationError,
    RecoveryOption,
    calculate_estimated_savings,
    calculate_gross_disruption_loss,
    calculate_recovery_plan_total_cost,
    evaluate_recovery_option,
    optimize_recovery_plan,
)


class TestFinancialOptimization(unittest.TestCase):
    """Comprehensive test suite for the Financial Impact & Recovery Engine."""

    def setUp(self) -> None:
        """Sets up reusable test scenarios and candidate recovery options."""
        self.standard_scenario = DisruptionScenario(
            scenario_id="SCEN-TEST-001",
            affected_quantity=100.0,
            loss_per_unit=250.0,
            disrupted_node="Processor X",
            available_budget=30000.0,
            max_lead_time_days=14.0,
            description="Lithium-ion battery supply disruption at Processor X",
        )

        self.option_within_budget = RecoveryOption(
            option_id="OPT-STANDARD-EXPEDITE",
            option_type="expedite_air_freight",
            description="Air expedite battery cells from secondary hub",
            recovery_cost=12000.0,
            available_capacity=100.0,
            lead_time_days=7.0,
        )

        self.option_over_budget = RecoveryOption(
            option_id="OPT-EXPENSIVE-REROUTE",
            option_type="charter_flight",
            description="Dedicated charter cargo flight",
            recovery_cost=45000.0,
            available_capacity=120.0,
            lead_time_days=3.0,
        )

        self.option_under_capacity = RecoveryOption(
            option_id="OPT-PARTIAL-BUFFER",
            option_type="local_buffer",
            description="Release localized emergency buffer stock",
            recovery_cost=5000.0,
            available_capacity=60.0,
            lead_time_days=2.0,
        )

        self.option_slow_lead_time = RecoveryOption(
            option_id="OPT-OCEAN-FREIGHT",
            option_type="ocean_reroute",
            description="Standard maritime freight re-routing",
            recovery_cost=4000.0,
            available_capacity=150.0,
            lead_time_days=28.0,
        )

    # =========================================================================
    # Test 1: Correct calculation of gross disruption loss
    # =========================================================================
    def test_01_calculate_gross_disruption_loss(self) -> None:
        """Verifies Gross Disruption Loss = Affected Quantity * Loss Per Unit."""
        # 100 units at $250/unit = $25,000.00
        loss = calculate_gross_disruption_loss(100.0, 250.0)
        self.assertEqual(loss, 25000.00)

        # Zero affected units results in $0 loss
        self.assertEqual(calculate_gross_disruption_loss(0.0, 500.0), 0.0)

        # Decimal precision test
        loss_decimal = calculate_gross_disruption_loss(12.5, 33.33)
        self.assertEqual(loss_decimal, 416.62)

    # =========================================================================
    # Test 2: Correct calculation of recovery plan total cost
    # =========================================================================
    def test_02_calculate_recovery_plan_total_cost(self) -> None:
        """Verifies Recovery Plan Total Cost = Recovery Cost + Residual Disruption Loss."""
        # $12,000 recovery cost + $0 residual loss = $12,000.00
        cost_full = calculate_recovery_plan_total_cost(12000.0, 0.0)
        self.assertEqual(cost_full, 12000.00)

        # $8,000 recovery cost + $5,000 residual loss = $13,000.00
        cost_partial = calculate_recovery_plan_total_cost(8000.0, 5000.0)
        self.assertEqual(cost_partial, 13000.00)

    # =========================================================================
    # Test 3: Correct calculation of estimated savings
    # =========================================================================
    def test_03_calculate_estimated_savings(self) -> None:
        """Verifies Estimated Savings = Baseline Disruption Loss - Recovery Plan Total Cost."""
        baseline = 25000.0
        plan_cost = 12000.0
        savings = calculate_estimated_savings(baseline, plan_cost)
        self.assertEqual(savings, 13000.00)

        # Negative savings when recovery costs more than baseline loss
        expensive_plan = 30000.0
        neg_savings = calculate_estimated_savings(baseline, expensive_plan)
        self.assertEqual(neg_savings, -5000.00)

    # =========================================================================
    # Test 4: Recovery option within the budget
    # =========================================================================
    def test_04_recovery_option_within_budget(self) -> None:
        """Verifies that an option within budget is deemed feasible."""
        evaluation = evaluate_recovery_option(self.option_within_budget, self.standard_scenario)
        self.assertTrue(evaluation["is_feasible"])
        self.assertEqual(len(evaluation["rejection_reasons"]), 0)
        self.assertEqual(evaluation["recovery_cost"], 12000.0)

    # =========================================================================
    # Test 5: Recovery option exceeding the budget
    # =========================================================================
    def test_05_recovery_option_exceeding_budget(self) -> None:
        """Verifies that an option exceeding budget is rejected with an explanation."""
        evaluation = evaluate_recovery_option(self.option_over_budget, self.standard_scenario)
        self.assertFalse(evaluation["is_feasible"])
        self.assertTrue(
            any("Budget overrun" in r for r in evaluation["rejection_reasons"]),
            msg=f"Expected Budget overrun in reasons, got: {evaluation['rejection_reasons']}",
        )

    # =========================================================================
    # Test 6: Recovery option exceeding available capacity (shortfall)
    # =========================================================================
    def test_06_recovery_option_exceeding_available_capacity(self) -> None:
        """Verifies that an option with capacity less than required quantity is rejected."""
        # Demanded = 100 units, available = 60 units
        evaluation = evaluate_recovery_option(self.option_under_capacity, self.standard_scenario)
        self.assertFalse(evaluation["is_feasible"])
        self.assertTrue(
            any("Capacity shortfall" in r for r in evaluation["rejection_reasons"]),
            msg=f"Expected Capacity shortfall in reasons, got: {evaluation['rejection_reasons']}",
        )

    # =========================================================================
    # Test 7: Recovery option violating a specified lead-time limit
    # =========================================================================
    def test_07_recovery_option_violating_lead_time_limit(self) -> None:
        """Verifies that an option exceeding max lead time is rejected with explanation."""
        # Max acceptable = 14 days, ocean freight = 28 days
        evaluation = evaluate_recovery_option(self.option_slow_lead_time, self.standard_scenario)
        self.assertFalse(evaluation["is_feasible"])
        self.assertTrue(
            any("Lead time violation" in r for r in evaluation["rejection_reasons"]),
            msg=f"Expected Lead time violation in reasons, got: {evaluation['rejection_reasons']}",
        )

    # =========================================================================
    # Test 8: Selection of the lowest-total-cost feasible option
    # =========================================================================
    def test_08_selection_of_lowest_total_cost_feasible_option(self) -> None:
        """Verifies that the engine selects the feasible option minimizing total cost."""
        opt_a = RecoveryOption(
            option_id="OPT-A",
            option_type="type_a",
            description="Supplier A option",
            recovery_cost=15000.0,
            available_capacity=100.0,
            lead_time_days=5.0,
        )
        opt_b = RecoveryOption(
            option_id="OPT-B",
            option_type="type_b",
            description="Supplier B option (cheaper)",
            recovery_cost=9500.0,
            available_capacity=100.0,
            lead_time_days=7.0,
        )
        opt_c = RecoveryOption(
            option_id="OPT-C",
            option_type="type_c",
            description="Supplier C option (higher cost)",
            recovery_cost=18000.0,
            available_capacity=100.0,
            lead_time_days=4.0,
        )

        result = optimize_recovery_plan(
            scenario=self.standard_scenario,
            recovery_options=[opt_a, opt_b, opt_c],
        )

        self.assertEqual(result["recommended_option_ids"], ["OPT-B"])
        self.assertEqual(result["recovery_cost"], 9500.0)
        self.assertEqual(result["estimated_total_cost"], 9500.0)
        self.assertEqual(result["estimated_savings"], 15500.0)  # 25000 - 9500 = 15500
        self.assertFalse(result["requires_human_approval"])

    # =========================================================================
    # Test 9: No feasible option available
    # =========================================================================
    def test_09_no_feasible_option_available(self) -> None:
        """Verifies engine behavior when all options violate constraints."""
        all_infeasible = [
            self.option_over_budget,
            self.option_under_capacity,
            self.option_slow_lead_time,
        ]
        result = optimize_recovery_plan(
            scenario=self.standard_scenario,
            recovery_options=all_infeasible,
        )

        self.assertEqual(result["recommended_option_ids"], [])
        self.assertEqual(result["recovery_cost"], 0.0)
        self.assertEqual(result["residual_disruption_loss"], self.standard_scenario.baseline_disruption_loss)
        self.assertEqual(result["estimated_total_cost"], self.standard_scenario.baseline_disruption_loss)
        self.assertEqual(result["estimated_savings"], 0.0)
        self.assertTrue(result["requires_human_approval"])
        self.assertIn("No feasible recovery plan was found", result["recommendation_explanation"])

    # =========================================================================
    # Test 10: Missing financial data is reported
    # =========================================================================
    def test_10_missing_financial_data_reported(self) -> None:
        """Verifies that missing capacity or lead time is not assumed 0, and is reported."""
        incomplete_option = {
            "option_id": "OPT-INCOMPLETE",
            "option_type": "third_party",
            "description": "Supplier with unknown capacity",
            "recovery_cost": 8000.0,
            "available_capacity": None,  # Missing capacity!
            "lead_time_days": 5.0,
        }

        result = optimize_recovery_plan(
            scenario=self.standard_scenario,
            recovery_options=[incomplete_option],
        )

        # Must NOT be deemed feasible
        self.assertEqual(result["recommended_option_ids"], [])
        self.assertTrue(len(result["missing_information"]) > 0)
        self.assertTrue(
            any("available_capacity" in info for info in result["missing_information"]),
            msg=f"Expected missing capacity warning, got: {result['missing_information']}",
        )

    # =========================================================================
    # Test 11: Invalid negative quantities or costs are handled appropriately
    # =========================================================================
    def test_11_invalid_negative_quantities_or_costs(self) -> None:
        """Verifies that negative inputs raise FinancialValidationError."""
        # Negative affected quantity
        with self.assertRaises(FinancialValidationError):
            DisruptionScenario(
                scenario_id="SCEN-NEG",
                affected_quantity=-50.0,
                loss_per_unit=100.0,
            )

        # Negative loss per unit
        with self.assertRaises(FinancialValidationError):
            calculate_gross_disruption_loss(100.0, -25.0)

        # Negative recovery cost
        with self.assertRaises(FinancialValidationError):
            RecoveryOption(
                option_id="OPT-NEG-COST",
                option_type="test",
                description="Negative cost",
                recovery_cost=-500.0,
                available_capacity=100.0,
                lead_time_days=5.0,
            )

        # Negative budget
        with self.assertRaises(FinancialValidationError):
            DisruptionScenario(
                scenario_id="SCEN-NEG-BUDGET",
                affected_quantity=10.0,
                loss_per_unit=10.0,
                available_budget=-1000.0,
            )

    # =========================================================================
    # Test 12: No double-counting of financial losses
    # =========================================================================
    def test_12_no_double_counting_of_financial_losses(self) -> None:
        """Verifies that baseline disruption loss is not added into plan total cost."""
        opt = RecoveryOption(
            option_id="OPT-FULL-COVERAGE",
            option_type="expedite",
            description="Full coverage expedite",
            recovery_cost=10000.0,
            available_capacity=100.0,
            lead_time_days=5.0,
            estimated_residual_loss=0.0,
        )

        result = optimize_recovery_plan(
            scenario=self.standard_scenario,
            recovery_options=[opt],
        )

        # Total cost MUST equal recovery_cost + residual_loss ($10,000 + $0 = $10,000)
        # It must NOT be $25,000 + $10,000 = $35,000!
        self.assertEqual(result["estimated_total_cost"], 10000.0)
        self.assertEqual(result["recovery_cost"], 10000.0)
        self.assertEqual(result["residual_disruption_loss"], 0.0)
        # Savings must be $25,000 - $10,000 = $15,000
        self.assertEqual(result["estimated_savings"], 15000.0)

    # =========================================================================
    # Test 13: No-action baseline comparison
    # =========================================================================
    def test_13_no_action_baseline_comparison(self) -> None:
        """Verifies that no-action baseline metrics are accurately calculated and reported."""
        result = optimize_recovery_plan(
            scenario=self.standard_scenario,
            recovery_options=[],  # No options provided
        )

        self.assertEqual(result["baseline_disruption_loss"], 25000.0)
        self.assertEqual(result["recovery_cost"], 0.0)
        self.assertEqual(result["residual_disruption_loss"], 25000.0)
        self.assertEqual(result["estimated_total_cost"], 25000.0)
        self.assertEqual(result["estimated_savings"], 0.0)
        self.assertEqual(result["recommended_option_ids"], [])
        self.assertTrue(result["requires_human_approval"])

    # =========================================================================
    # Test 14: Correct structured output schema
    # =========================================================================
    def test_14_correct_structured_output_schema(self) -> None:
        """Verifies that the output dictionary adheres strictly to the required schema."""
        result = optimize_recovery_plan(
            scenario=self.standard_scenario,
            recovery_options=[self.option_within_budget, self.option_over_budget],
        )

        required_keys = [
            "scenario_id",
            "baseline_disruption_loss",
            "available_budget",
            "recommended_option_ids",
            "recovery_cost",
            "residual_disruption_loss",
            "estimated_total_cost",
            "estimated_savings",
            "feasible_options",
            "rejected_options",
            "assumptions",
            "missing_information",
            "recommendation_explanation",
            "requires_human_approval",
        ]

        for key in required_keys:
            self.assertIn(key, result, f"Missing required output key: '{key}'")

        # Check types
        self.assertIsInstance(result["scenario_id"], str)
        self.assertIsInstance(result["baseline_disruption_loss"], float)
        self.assertIsInstance(result["recommended_option_ids"], list)
        self.assertIsInstance(result["recovery_cost"], float)
        self.assertIsInstance(result["residual_disruption_loss"], float)
        self.assertIsInstance(result["estimated_total_cost"], float)
        self.assertIsInstance(result["estimated_savings"], float)
        self.assertIsInstance(result["feasible_options"], list)
        self.assertIsInstance(result["rejected_options"], list)
        self.assertIsInstance(result["assumptions"], list)
        self.assertIsInstance(result["missing_information"], list)
        self.assertIsInstance(result["recommendation_explanation"], str)
        self.assertIsInstance(result["requires_human_approval"], bool)

    # =========================================================================
    # Additional Test: Multi-Option Combination Optimization
    # =========================================================================
    def test_15_multi_option_combination(self) -> None:
        """Verifies multi-supplier allocation when two partial options together fulfill demand."""
        # Demand is 100 units. Neither option alone has 100 capacity.
        opt_part1 = RecoveryOption(
            option_id="OPT-PART-1",
            option_type="supplier_split",
            description="Supplier 1 partial batch",
            recovery_cost=6000.0,
            available_capacity=60.0,
            lead_time_days=6.0,
        )
        opt_part2 = RecoveryOption(
            option_id="OPT-PART-2",
            option_type="supplier_split",
            description="Supplier 2 partial batch",
            recovery_cost=5000.0,
            available_capacity=50.0,
            lead_time_days=8.0,
        )

        result = optimize_recovery_plan(
            scenario=self.standard_scenario,
            recovery_options=[opt_part1, opt_part2],
            allow_multi_option=True,
        )

        # Combined capacity = 60 + 50 = 110 >= 100 units
        # Combined cost = 6000 + 5000 = 11000 <= 30000 budget
        self.assertCountEqual(result["recommended_option_ids"], ["OPT-PART-1", "OPT-PART-2"])
        self.assertEqual(result["recovery_cost"], 11000.0)
        self.assertEqual(result["residual_disruption_loss"], 0.0)
        self.assertEqual(result["estimated_total_cost"], 11000.0)
        self.assertEqual(result["estimated_savings"], 14000.0)  # 25000 - 11000


if __name__ == "__main__":
    unittest.main()

