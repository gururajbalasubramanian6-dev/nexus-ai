"""
Nexus-AI: Financial Impact & Recovery Optimization Engine
Member 4 Component

This module provides deterministic financial impact evaluation and recovery
optimization for multi-tier supply chain disruptions.

Key capabilities:
1. Gross Disruption Loss Calculation:
   Quantifies baseline business loss from disrupted units and per-unit loss.
2. Feasibility Constraint Evaluation:
   Validates candidate recovery options against budget limits, capacity requirements,
   lead-time tolerances, and data completeness.
3. Total Cost & Savings Analysis:
   Evaluates recovery expenditures versus residual disruption losses to determine
   net economic benefit over taking no action.
4. Deterministic Optimization:
   Selects the optimal recovery option (or feasible combination) minimizing
   total estimated financial exposure.
5. Structured Decision Explanation:
   Produces comprehensive, auditable outputs ready for Streamlit dashboard integration (Member 5).
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple, Union


class FinancialValidationError(ValueError):
    """Raised when financial inputs, quantities, or constraints are invalid or inconsistent."""
    pass


@dataclass
class DisruptionScenario:
    """
    Represents an active or hypothetical supply chain disruption scenario.

    Attributes:
        scenario_id: Unique identifier for the disruption event.
        affected_quantity: Volume of product or raw material disrupted (must be >= 0).
        loss_per_unit: Estimated gross financial loss incurred per unfulfilled unit (must be >= 0).
        disrupted_node: Optional name of the impacted supply chain facility/supplier.
        available_budget: Maximum capital allocated for recovery (must be >= 0 if provided).
        max_lead_time_days: Maximum acceptable lead time in days (must be >= 0 if provided).
        baseline_disruption_loss: Pre-computed baseline loss. If omitted, computed as
            affected_quantity * loss_per_unit.
        description: Brief contextual description of the disruption.
        currency: ISO currency code (default: 'USD').
        additional_metadata: Optional dictionary for upstream/downstream agent context.
    """
    scenario_id: str
    affected_quantity: float
    loss_per_unit: float
    disrupted_node: Optional[str] = None
    available_budget: Optional[float] = None
    max_lead_time_days: Optional[float] = None
    baseline_disruption_loss: Optional[float] = None
    description: str = ""
    currency: str = "USD"
    additional_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.scenario_id or not str(self.scenario_id).strip():
            raise FinancialValidationError("scenario_id must be a non-empty string.")

        try:
            self.affected_quantity = float(self.affected_quantity)
        except (TypeError, ValueError) as err:
            raise FinancialValidationError(
                f"affected_quantity must be numeric, got: {self.affected_quantity}"
            ) from err

        if self.affected_quantity < 0:
            raise FinancialValidationError(
                f"affected_quantity cannot be negative: {self.affected_quantity}"
            )

        try:
            self.loss_per_unit = float(self.loss_per_unit)
        except (TypeError, ValueError) as err:
            raise FinancialValidationError(
                f"loss_per_unit must be numeric, got: {self.loss_per_unit}"
            ) from err

        if self.loss_per_unit < 0:
            raise FinancialValidationError(
                f"loss_per_unit cannot be negative: {self.loss_per_unit}"
            )

        if self.available_budget is not None:
            try:
                self.available_budget = float(self.available_budget)
            except (TypeError, ValueError) as err:
                raise FinancialValidationError(
                    f"available_budget must be numeric, got: {self.available_budget}"
                ) from err
            if self.available_budget < 0:
                raise FinancialValidationError(
                    f"available_budget cannot be negative: {self.available_budget}"
                )

        if self.max_lead_time_days is not None:
            try:
                self.max_lead_time_days = float(self.max_lead_time_days)
            except (TypeError, ValueError) as err:
                raise FinancialValidationError(
                    f"max_lead_time_days must be numeric, got: {self.max_lead_time_days}"
                ) from err
            if self.max_lead_time_days < 0:
                raise FinancialValidationError(
                    f"max_lead_time_days cannot be negative: {self.max_lead_time_days}"
                )

        if self.baseline_disruption_loss is not None:
            try:
                self.baseline_disruption_loss = float(self.baseline_disruption_loss)
            except (TypeError, ValueError) as err:
                raise FinancialValidationError(
                    f"baseline_disruption_loss must be numeric, got: {self.baseline_disruption_loss}"
                ) from err
            if self.baseline_disruption_loss < 0:
                raise FinancialValidationError(
                    f"baseline_disruption_loss cannot be negative: {self.baseline_disruption_loss}"
                )
        else:
            self.baseline_disruption_loss = calculate_gross_disruption_loss(
                self.affected_quantity, self.loss_per_unit
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes scenario to a standard Python dictionary."""
        return asdict(self)


@dataclass
class RecoveryOption:
    """
    Represents a candidate recovery or mitigation action for a disruption.

    Attributes:
        option_id: Unique identifier for the recovery option (e.g., 'OPT-EXPEDITE-01').
        option_type: Category of action (e.g., 'expedite', 'alternate_supplier', 'buffer_inventory').
        description: Readable summary of the proposed recovery action.
        recovery_cost: Total financial expenditure required to implement this option.
        available_capacity: Number of units this option can supply or recover.
        lead_time_days: Number of days required to deliver the recovery action.
        required_quantity: Quantity demanded for this specific option. If None, inherits
            from the scenario's affected_quantity during evaluation.
        maximum_acceptable_lead_time_days: Option-specific lead-time ceiling. If None,
            inherits from scenario's max_lead_time_days.
        estimated_residual_loss: Explicitly supplied remaining loss after this action.
            If None, computed deterministically based on unfulfilled quantity and loss_per_unit.
        supplier_name: Optional name of the alternative supplier or provider.
        additional_metadata: Optional dictionary for tracking custom logistics parameters.
    """
    option_id: str
    option_type: str
    description: str
    recovery_cost: float
    available_capacity: float
    lead_time_days: float
    required_quantity: Optional[float] = None
    maximum_acceptable_lead_time_days: Optional[float] = None
    estimated_residual_loss: Optional[float] = None
    supplier_name: Optional[str] = None
    additional_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.option_id or not str(self.option_id).strip():
            raise FinancialValidationError("option_id must be a non-empty string.")

        if not self.option_type or not str(self.option_type).strip():
            raise FinancialValidationError("option_type must be a non-empty string.")

        try:
            self.recovery_cost = float(self.recovery_cost)
        except (TypeError, ValueError) as err:
            raise FinancialValidationError(
                f"recovery_cost must be numeric for option '{self.option_id}', got: {self.recovery_cost}"
            ) from err

        if self.recovery_cost < 0:
            raise FinancialValidationError(
                f"recovery_cost cannot be negative for option '{self.option_id}': {self.recovery_cost}"
            )

        if self.available_capacity is not None:
            try:
                self.available_capacity = float(self.available_capacity)
            except (TypeError, ValueError) as err:
                raise FinancialValidationError(
                    f"available_capacity must be numeric for option '{self.option_id}', got: {self.available_capacity}"
                ) from err
            if self.available_capacity < 0:
                raise FinancialValidationError(
                    f"available_capacity cannot be negative for option '{self.option_id}': {self.available_capacity}"
                )

        if self.lead_time_days is not None:
            try:
                self.lead_time_days = float(self.lead_time_days)
            except (TypeError, ValueError) as err:
                raise FinancialValidationError(
                    f"lead_time_days must be numeric for option '{self.option_id}', got: {self.lead_time_days}"
                ) from err
            if self.lead_time_days < 0:
                raise FinancialValidationError(
                    f"lead_time_days cannot be negative for option '{self.option_id}': {self.lead_time_days}"
                )

        if self.required_quantity is not None:
            try:
                self.required_quantity = float(self.required_quantity)
            except (TypeError, ValueError) as err:
                raise FinancialValidationError(
                    f"required_quantity must be numeric for option '{self.option_id}', got: {self.required_quantity}"
                ) from err
            if self.required_quantity < 0:
                raise FinancialValidationError(
                    f"required_quantity cannot be negative for option '{self.option_id}': {self.required_quantity}"
                )

        if self.maximum_acceptable_lead_time_days is not None:
            try:
                self.maximum_acceptable_lead_time_days = float(self.maximum_acceptable_lead_time_days)
            except (TypeError, ValueError) as err:
                raise FinancialValidationError(
                    f"maximum_acceptable_lead_time_days must be numeric for option '{self.option_id}', "
                    f"got: {self.maximum_acceptable_lead_time_days}"
                ) from err
            if self.maximum_acceptable_lead_time_days < 0:
                raise FinancialValidationError(
                    f"maximum_acceptable_lead_time_days cannot be negative for option '{self.option_id}': "
                    f"{self.maximum_acceptable_lead_time_days}"
                )

        if self.estimated_residual_loss is not None:
            try:
                self.estimated_residual_loss = float(self.estimated_residual_loss)
            except (TypeError, ValueError) as err:
                raise FinancialValidationError(
                    f"estimated_residual_loss must be numeric for option '{self.option_id}', "
                    f"got: {self.estimated_residual_loss}"
                ) from err
            if self.estimated_residual_loss < 0:
                raise FinancialValidationError(
                    f"estimated_residual_loss cannot be negative for option '{self.option_id}': "
                    f"{self.estimated_residual_loss}"
                )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes option to a standard Python dictionary."""
        return asdict(self)


# ============================================================================
# Core Financial Calculation Functions (Step 3: Deterministic & Transparent)
# ============================================================================

def calculate_gross_disruption_loss(affected_quantity: float, loss_per_unit: float) -> float:
    """
    Calculates the unmitigated gross business loss resulting from a disruption.

    Formula:
        Gross disruption loss = affected_quantity * loss_per_unit

    Assumptions & Units:
        - affected_quantity in units (pieces, kg, batches, etc.). Must be >= 0.
        - loss_per_unit in currency units per product unit (e.g., USD/unit). Must be >= 0.
        - Result is in currency units (USD).

    Parameters:
        affected_quantity: Number of unfulfilled or disrupted units.
        loss_per_unit: Revenue or operating loss per disrupted unit.

    Returns:
        float: Calculated gross loss rounded to 2 decimal places.

    Raises:
        FinancialValidationError: If either value is None, non-numeric, or negative.
    """
    if affected_quantity is None or loss_per_unit is None:
        raise FinancialValidationError("affected_quantity and loss_per_unit must not be None.")

    try:
        qty = float(affected_quantity)
        rate = float(loss_per_unit)
    except (TypeError, ValueError) as err:
        raise FinancialValidationError(
            f"Inputs to calculate_gross_disruption_loss must be numeric. Got qty={affected_quantity}, rate={loss_per_unit}"
        ) from err

    if qty < 0:
        raise FinancialValidationError(f"affected_quantity cannot be negative: {qty}")
    if rate < 0:
        raise FinancialValidationError(f"loss_per_unit cannot be negative: {rate}")

    return round(qty * rate, 2)


def calculate_recovery_plan_total_cost(recovery_cost: float, residual_loss: float) -> float:
    """
    Calculates the total financial exposure of adopting a recovery plan.

    Formula:
        Recovery plan total cost = Recovery cost + Remaining disruption loss

    Assumptions & Units:
        - Recovery cost: Direct expenses paid to suppliers, logistics carriers, or overtime.
        - Residual loss: Unmitigated disruption loss still suffered due to partial fulfillment.
        - Avoids double counting: baseline loss is NOT added here; only the remaining portion.

    Parameters:
        recovery_cost: Expenditure needed to execute the recovery action.
        residual_loss: Remaining disruption loss after the recovery action.

    Returns:
        float: Combined total cost rounded to 2 decimal places.

    Raises:
        FinancialValidationError: If inputs are missing, non-numeric, or negative.
    """
    if recovery_cost is None or residual_loss is None:
        raise FinancialValidationError("recovery_cost and residual_loss must not be None.")

    try:
        cost = float(recovery_cost)
        residual = float(residual_loss)
    except (TypeError, ValueError) as err:
        raise FinancialValidationError(
            f"Inputs to calculate_recovery_plan_total_cost must be numeric. Got cost={recovery_cost}, residual={residual_loss}"
        ) from err

    if cost < 0:
        raise FinancialValidationError(f"recovery_cost cannot be negative: {cost}")
    if residual < 0:
        raise FinancialValidationError(f"residual_loss cannot be negative: {residual}")

    return round(cost + residual, 2)


def calculate_estimated_savings(baseline_disruption_loss: float, recovery_plan_total_cost: float) -> float:
    """
    Calculates estimated net financial savings compared to taking no recovery action.

    Formula:
        Estimated savings = Baseline disruption loss - Recovery plan total cost

    Interpretation:
        - Savings > 0: The recovery plan preserves capital and is economically advantageous.
        - Savings = 0: Break-even (cost of recovery equals baseline disruption loss).
        - Savings < 0: The recovery action costs more than the disruption it prevents.

    Parameters:
        baseline_disruption_loss: Unmitigated gross disruption loss (no action taken).
        recovery_plan_total_cost: Total cost of executing the recovery plan plus residual loss.

    Returns:
        float: Net savings rounded to 2 decimal places.

    Raises:
        FinancialValidationError: If inputs are missing, non-numeric, or negative.
    """
    if baseline_disruption_loss is None or recovery_plan_total_cost is None:
        raise FinancialValidationError("baseline_disruption_loss and recovery_plan_total_cost must not be None.")

    try:
        base_loss = float(baseline_disruption_loss)
        plan_cost = float(recovery_plan_total_cost)
    except (TypeError, ValueError) as err:
        raise FinancialValidationError(
            f"Inputs to calculate_estimated_savings must be numeric. Got base_loss={baseline_disruption_loss}, plan_cost={recovery_plan_total_cost}"
        ) from err

    if base_loss < 0:
        raise FinancialValidationError(f"baseline_disruption_loss cannot be negative: {base_loss}")
    if plan_cost < 0:
        raise FinancialValidationError(f"recovery_plan_total_cost cannot be negative: {plan_cost}")

    return round(base_loss - plan_cost, 2)


# ============================================================================
# Option Feasibility & Constraint Evaluation (Step 4)
# ============================================================================

def evaluate_recovery_option(
    option: Union[RecoveryOption, Dict[str, Any]],
    scenario: Union[DisruptionScenario, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluates a candidate recovery option against all business and operational constraints.

    Constraints evaluated:
    1. Budget Constraint:
       Recovery cost must not exceed available budget (if budget is specified).
    2. Capacity Constraint:
       Available capacity must cover assigned/required quantity. Missing capacity is
       flagged as incomplete data and deemed infeasible.
    3. Lead-Time Constraint:
       Lead time must not exceed maximum acceptable lead time (if specified on option
       or scenario). Missing lead-time when required is deemed infeasible.
    4. Data Integrity:
       Negative values, non-numeric entries, or missing essential identifiers are rejected.

    Parameters:
        option: RecoveryOption instance or equivalent dictionary.
        scenario: DisruptionScenario instance or equivalent dictionary.

    Returns:
        Dict[str, Any]: Detailed evaluation record containing:
            - option_id: str
            - option_type: str
            - description: str
            - is_feasible: bool
            - rejection_reasons: List[str]
            - recovery_cost: float
            - available_capacity: Optional[float]
            - required_quantity: float
            - lead_time_days: Optional[float]
            - max_lead_time_days: Optional[float]
            - residual_loss: float
            - total_estimated_cost: float
            - estimated_savings: float
            - missing_fields: List[str]
    """
    # Coerce to dataclasses if dictionaries were passed
    if isinstance(scenario, dict):
        scen = DisruptionScenario(**scenario)
    else:
        scen = scenario

    if isinstance(option, dict):
        # Handle cases where capacity or lead_time might be None in dict
        opt_data = dict(option)
        opt = RecoveryOption(**opt_data)
    else:
        opt = option

    rejection_reasons: List[str] = []
    missing_fields: List[str] = []

    # Determine assigned required quantity
    target_quantity = (
        opt.required_quantity if opt.required_quantity is not None else scen.affected_quantity
    )

    # 1. Capacity evaluation
    if opt.available_capacity is None:
        missing_fields.append("available_capacity")
        rejection_reasons.append(
            f"Missing available capacity data for option '{opt.option_id}'. "
            "Cannot verify feasibility without capacity information."
        )
    elif opt.available_capacity < target_quantity:
        shortfall = round(target_quantity - opt.available_capacity, 2)
        rejection_reasons.append(
            f"Capacity shortfall: available capacity ({opt.available_capacity}) is less than "
            f"required quantity ({target_quantity}). Deficit: {shortfall} units."
        )

    # 2. Budget evaluation
    if scen.available_budget is not None:
        if opt.recovery_cost > scen.available_budget:
            overage = round(opt.recovery_cost - scen.available_budget, 2)
            rejection_reasons.append(
                f"Budget overrun: recovery cost (${opt.recovery_cost:,.2f}) exceeds "
                f"available budget (${scen.available_budget:,.2f}) by ${overage:,.2f}."
            )

    # 3. Lead-time evaluation
    effective_max_lead_time = (
        opt.maximum_acceptable_lead_time_days
        if opt.maximum_acceptable_lead_time_days is not None
        else scen.max_lead_time_days
    )

    if effective_max_lead_time is not None:
        if opt.lead_time_days is None:
            missing_fields.append("lead_time_days")
            rejection_reasons.append(
                f"Missing lead time data for option '{opt.option_id}' while maximum acceptable "
                f"lead time is {effective_max_lead_time} days."
            )
        elif opt.lead_time_days > effective_max_lead_time:
            excess_days = round(opt.lead_time_days - effective_max_lead_time, 1)
            rejection_reasons.append(
                f"Lead time violation: lead time ({opt.lead_time_days} days) exceeds maximum "
                f"acceptable lead time ({effective_max_lead_time} days) by {excess_days} days."
            )

    # Determine residual disruption loss
    if opt.estimated_residual_loss is not None:
        residual_loss = round(float(opt.estimated_residual_loss), 2)
    else:
        # Deterministically compute residual loss if capacity is known
        if opt.available_capacity is not None:
            recovered_units = min(opt.available_capacity, target_quantity)
            unrecovered_units = max(0.0, target_quantity - recovered_units)
            residual_loss = calculate_gross_disruption_loss(unrecovered_units, scen.loss_per_unit)
        else:
            # Capacity missing: cannot safely claim residual loss is 0
            residual_loss = scen.baseline_disruption_loss
            missing_fields.append("residual_loss_due_to_missing_capacity")

    total_cost = calculate_recovery_plan_total_cost(opt.recovery_cost, residual_loss)
    savings = calculate_estimated_savings(scen.baseline_disruption_loss, total_cost)

    is_feasible = (len(rejection_reasons) == 0)

    return {
        "option_id": opt.option_id,
        "option_type": opt.option_type,
        "description": opt.description,
        "supplier_name": opt.supplier_name,
        "is_feasible": is_feasible,
        "rejection_reasons": rejection_reasons,
        "recovery_cost": opt.recovery_cost,
        "available_capacity": opt.available_capacity,
        "required_quantity": target_quantity,
        "lead_time_days": opt.lead_time_days,
        "max_acceptable_lead_time_days": effective_max_lead_time,
        "residual_disruption_loss": residual_loss,
        "estimated_total_cost": total_cost,
        "estimated_savings": savings,
        "missing_fields": missing_fields,
        "additional_metadata": opt.additional_metadata,
    }


# ============================================================================
# Optimization Engine (Step 5 & Step 6)
# ============================================================================

def optimize_recovery_plan(
    scenario: Union[DisruptionScenario, Dict[str, Any]],
    recovery_options: List[Union[RecoveryOption, Dict[str, Any]]],
    allow_multi_option: bool = False,
) -> Dict[str, Any]:
    """
    Evaluates candidate recovery options and optimizes the recovery plan to minimize
    total estimated financial exposure.

    Optimization Objective:
        Minimize Total Estimated Cost = Recovery Expenditure + Residual Disruption Loss

    Constraints:
        1. Recovery cost <= available budget (if provided).
        2. Recovery capacity >= required disrupted quantity.
        3. Lead time <= max acceptable lead time (if specified).
        4. Complete data verification.

    Single-Option vs Multi-Option Selection:
        - By default (allow_multi_option=False), selects the single best feasible option.
        - If allow_multi_option=True and no single option covers 100% capacity, evaluates
          minimal combinations of options that together fulfill the required quantity
          within budget and lead-time constraints.

    Baseline Comparison:
        Always compares the recommended plan against the No-Action Baseline:
        - No-Action Recovery Expenditure = $0.00
        - No-Action Residual Loss = Baseline Disruption Loss
        - No-Action Total Cost = Baseline Disruption Loss

    Parameters:
        scenario: DisruptionScenario instance or dict of scenario fields.
        recovery_options: List of RecoveryOption instances or option dicts.
        allow_multi_option: Boolean flag allowing multi-supplier/option combination if needed.

    Returns:
        Dict[str, Any]: Structured output adhering to the project schema:
            - scenario_id: str
            - baseline_disruption_loss: float
            - available_budget: Optional[float]
            - recommended_option_ids: List[str]
            - recovery_cost: float
            - residual_disruption_loss: float
            - estimated_total_cost: float
            - estimated_savings: float
            - feasible_options: List[Dict[str, Any]]
            - rejected_options: List[Dict[str, Any]]
            - assumptions: List[str]
            - missing_information: List[str]
            - recommendation_explanation: str
            - requires_human_approval: bool
    """
    # 1. Ingest and validate scenario
    if isinstance(scenario, dict):
        scen = DisruptionScenario(**scenario)
    elif isinstance(scenario, DisruptionScenario):
        scen = scenario
    else:
        raise FinancialValidationError(
            f"scenario must be a DisruptionScenario or dict, got: {type(scenario)}"
        )

    assumptions: List[str] = [
        "Gross disruption loss calculated as affected_quantity * loss_per_unit.",
        "Total estimated cost equals recovery expenditure plus residual disruption loss.",
        "Estimated savings equals baseline disruption loss minus total recovery plan cost.",
        "No-action baseline assumes zero recovery expenditure and 100% disruption loss.",
    ]
    missing_info: List[str] = []

    if scen.available_budget is None:
        assumptions.append("No budget limit specified; recovery options evaluated without budget cap.")
    if scen.max_lead_time_days is None:
        assumptions.append("No maximum lead time constraint specified; lead times evaluated without ceiling.")

    # 2. Ingest and evaluate candidate options
    if not isinstance(recovery_options, list):
        raise FinancialValidationError("recovery_options must be a list of options or option dicts.")

    feasible_options: List[Dict[str, Any]] = []
    rejected_options: List[Dict[str, Any]] = []

    for idx, opt_input in enumerate(recovery_options):
        try:
            evaluated = evaluate_recovery_option(opt_input, scen)
            if evaluated["missing_fields"]:
                for field_name in evaluated["missing_fields"]:
                    missing_info.append(
                        f"Option '{evaluated['option_id']}': missing required field '{field_name}'."
                    )
            if evaluated["is_feasible"]:
                feasible_options.append(evaluated)
            else:
                rejected_options.append(evaluated)
        except FinancialValidationError as val_err:
            opt_id = (
                opt_input.get("option_id", f"OPTION_{idx}")
                if isinstance(opt_input, dict)
                else getattr(opt_input, "option_id", f"OPTION_{idx}")
            )
            rejected_options.append({
                "option_id": opt_id,
                "is_feasible": False,
                "rejection_reasons": [f"Validation Error: {str(val_err)}"],
                "recovery_cost": getattr(opt_input, "recovery_cost", None) if not isinstance(opt_input, dict) else opt_input.get("recovery_cost"),
                "estimated_total_cost": None,
                "estimated_savings": None,
                "missing_fields": ["validation_failed"],
            })
            missing_info.append(f"Option '{opt_id}' failed input validation: {str(val_err)}")

    # 3. Optimization Logic
    # Case A: Feasible single options exist
    if feasible_options:
        # Sort deterministic: lowest total cost first, tie-break by lower lead time, then lower recovery cost
        sorted_feasible = sorted(
            feasible_options,
            key=lambda x: (
                x["estimated_total_cost"],
                x["lead_time_days"] if x["lead_time_days"] is not None else float("inf"),
                x["recovery_cost"],
                x["option_id"],
            ),
        )
        best_single = sorted_feasible[0]

        rec_option_ids = [best_single["option_id"]]
        rec_cost = best_single["recovery_cost"]
        rec_residual = best_single["residual_disruption_loss"]
        rec_total = best_single["estimated_total_cost"]
        rec_savings = best_single["estimated_savings"]

        # Check if savings are negative (recovery costs more than taking no action)
        if rec_savings < 0:
            explanation = (
                f"Option '{best_single['option_id']}' ({best_single['description']}) is technically feasible "
                f"but economically sub-optimal. Its total cost of ${rec_total:,.2f} exceeds the "
                f"no-action baseline loss of ${scen.baseline_disruption_loss:,.2f} by ${abs(rec_savings):,.2f}. "
                "Adopting this recovery plan results in negative net savings. Taking no action or negotiating "
                "lower recovery costs is advised unless operational non-financial imperatives mandate recovery."
            )
            requires_human_approval = True
        elif rec_savings == 0:
            explanation = (
                f"Option '{best_single['option_id']}' breaks even with the no-action baseline at "
                f"${rec_total:,.2f}. Recommended if customer goodwill and service continuity are prioritized."
            )
            requires_human_approval = True
        else:
            explanation = (
                f"Recommended recovery option: '{best_single['option_id']}' ({best_single['description']}). "
                f"It satisfies all capacity ({best_single['available_capacity']} units), "
                f"budget (${rec_cost:,.2f} <= ${scen.available_budget:,.2f} if applicable), "
                f"and lead-time ({best_single['lead_time_days']} days) constraints. "
                f"Total plan cost is ${rec_total:,.2f}, delivering estimated net savings of "
                f"${rec_savings:,.2f} (a {round((rec_savings / scen.baseline_disruption_loss) * 100, 1)}% savings "
                f"over the no-action loss of ${scen.baseline_disruption_loss:,.2f})."
            )
            requires_human_approval = False

    # Case B: No single feasible option, but multi-option enabled
    elif allow_multi_option and len(recovery_options) > 1:
        multi_result = _evaluate_multi_option_combination(
            recovery_options=recovery_options,
            scenario=scen,
        )
        if multi_result is not None:
            rec_option_ids = multi_result["recommended_option_ids"]
            rec_cost = multi_result["recovery_cost"]
            rec_residual = multi_result["residual_disruption_loss"]
            rec_total = multi_result["estimated_total_cost"]
            rec_savings = multi_result["estimated_savings"]
            explanation = multi_result["explanation"]
            requires_human_approval = True  # Multi-supplier split always warrants operational confirmation
            assumptions.append(
                f"Quantities partitioned across {len(rec_option_ids)} distinct recovery options: {', '.join(rec_option_ids)}."
            )
        else:
            rec_option_ids = []
            rec_cost = 0.0
            rec_residual = scen.baseline_disruption_loss
            rec_total = scen.baseline_disruption_loss
            rec_savings = 0.0
            explanation = (
                "No feasible recovery option or combination found. All evaluated options violated budget, "
                "capacity, lead-time, or data integrity constraints. The platform recommends the No-Action "
                f"Baseline (Estimated Disruption Loss: ${scen.baseline_disruption_loss:,.2f})."
            )
            requires_human_approval = True

    # Case C: No feasible option available
    else:
        rec_option_ids = []
        rec_cost = 0.0
        rec_residual = scen.baseline_disruption_loss
        rec_total = scen.baseline_disruption_loss
        rec_savings = 0.0
        explanation = (
            "No feasible recovery plan was found. All candidate options violated either available budget, "
            "minimum capacity requirements, lead-time tolerances, or lacked required data. "
            f"Under the No-Action Baseline, estimated gross disruption loss remains ${scen.baseline_disruption_loss:,.2f}."
        )
        requires_human_approval = True

    # High budget utilization warning (> 85% of budget)
    if scen.available_budget is not None and rec_cost > 0:
        budget_ratio = rec_cost / scen.available_budget
        if budget_ratio >= 0.85:
            requires_human_approval = True
            assumptions.append(
                f"High budget utilization: recommended recovery plan utilizes {round(budget_ratio * 100, 1)}% "
                f"of available budget (${rec_cost:,.2f} / ${scen.available_budget:,.2f})."
            )

    return {
        "scenario_id": scen.scenario_id,
        "disrupted_node": scen.disrupted_node,
        "affected_quantity": scen.affected_quantity,
        "loss_per_unit": scen.loss_per_unit,
        "baseline_disruption_loss": scen.baseline_disruption_loss,
        "available_budget": scen.available_budget,
        "max_lead_time_days": scen.max_lead_time_days,
        "recommended_option_ids": rec_option_ids,
        "recovery_cost": rec_cost,
        "residual_disruption_loss": rec_residual,
        "estimated_total_cost": rec_total,
        "estimated_savings": rec_savings,
        "feasible_options": feasible_options,
        "rejected_options": rejected_options,
        "assumptions": assumptions,
        "missing_information": missing_info,
        "recommendation_explanation": explanation,
        "requires_human_approval": requires_human_approval,
    }


def _evaluate_multi_option_combination(
    recovery_options: List[Union[RecoveryOption, Dict[str, Any]]],
    scenario: DisruptionScenario,
) -> Optional[Dict[str, Any]]:
    """
    Evaluates feasible multi-option subsets where multiple suppliers or mitigations
    can be combined to meet the total required quantity.

    Returns the combination minimizing total estimated cost, or None if no combination is feasible.
    """
    parsed_options: List[RecoveryOption] = []
    for opt in recovery_options:
        try:
            if isinstance(opt, dict):
                parsed_options.append(RecoveryOption(**opt))
            else:
                parsed_options.append(opt)
        except Exception:
            continue

    if len(parsed_options) < 2:
        return None

    target_qty = scenario.affected_quantity
    best_combo_record: Optional[Dict[str, Any]] = None
    lowest_cost = float("inf")

    # Evaluate combinations of size 2 up to len(parsed_options)
    for r in range(2, len(parsed_options) + 1):
        for combo in combinations(parsed_options, r):
            total_capacity = sum(c.available_capacity for c in combo if c.available_capacity is not None)
            total_cost = sum(c.recovery_cost for c in combo)
            max_lead = max(c.lead_time_days for c in combo if c.lead_time_days is not None)

            # Check capacity
            if total_capacity < target_qty:
                continue

            # Check budget
            if scenario.available_budget is not None and total_cost > scenario.available_budget:
                continue

            # Check lead time
            if scenario.max_lead_time_days is not None and max_lead > scenario.max_lead_time_days:
                continue

            # Check individual lead time constraints if present
            lead_time_violated = False
            for c in combo:
                if c.maximum_acceptable_lead_time_days is not None and c.lead_time_days > c.maximum_acceptable_lead_time_days:
                    lead_time_violated = True
                    break
            if lead_time_violated:
                continue

            # Residual loss is 0 since combined capacity >= target_qty
            residual_loss = 0.0
            plan_total = calculate_recovery_plan_total_cost(total_cost, residual_loss)
            savings = calculate_estimated_savings(scenario.baseline_disruption_loss, plan_total)

            if plan_total < lowest_cost:
                lowest_cost = plan_total
                combo_ids = [c.option_id for c in combo]
                best_combo_record = {
                    "recommended_option_ids": combo_ids,
                    "recovery_cost": round(total_cost, 2),
                    "residual_disruption_loss": 0.0,
                    "estimated_total_cost": round(plan_total, 2),
                    "estimated_savings": round(savings, 2),
                    "explanation": (
                        f"Recommended multi-supplier recovery plan combining options: {', '.join(combo_ids)}. "
                        f"Combined capacity is {total_capacity} units (demand: {target_qty} units). "
                        f"Combined recovery cost is ${total_cost:,.2f} with maximum lead time of {max_lead} days, "
                        f"delivering estimated net savings of ${savings:,.2f}."
                    ),
                }

    return best_combo_record

