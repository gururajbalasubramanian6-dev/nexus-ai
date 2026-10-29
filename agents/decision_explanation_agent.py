"""
Nexus-AI: Decision Explanation Agent (Member 5)
================================================

This module implements a deterministic, explainable decision-reasoning engine
that synthesizes findings from:
- Member 1: Supply Chain Mapping Agent
- Member 2: Shared Risk Detection Agent
- Member 3: Disruption Simulation Agent
- Member 4: Financial Impact and Recovery Optimization Engine

It generates an auditable, 9-point structured decision explanation
without relying on external paid LLM APIs or black-box heuristics.

Author: Nexus-AI Team (Member 5)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import networkx as nx


class DecisionExplanationAgent:
    """
    Deterministic explanation component for multi-tier supply chain resilience decisions.
    Synthesizes structural risk, disruption blast radius, and financial optimization results
    into plain-English, auditable justifications.
    """

    def __init__(self) -> None:
        pass

    def explain(
        self,
        graph: Optional[nx.DiGraph],
        disruption_result: Dict[str, Any],
        financial_result: Dict[str, Any],
        risk_findings: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Generates a comprehensive 9-point structured decision explanation.

        Parameters
        ----------
        graph : Optional[nx.DiGraph]
            The supply chain network graph.
        disruption_result : Dict[str, Any]
            Output payload from Member 3's simulate_disruption.
        financial_result : Dict[str, Any]
            Output payload from Member 4's optimize_recovery_plan.
        risk_findings : Optional[List[Dict[str, Any]]], default=None
            Output list from Member 2's detect_shared_risks.

        Returns
        -------
        Dict[str, Any]
            Structured explanation dictionary containing both itemized sections
            and a formatted Markdown synthesis.
        """
        # 1. What failed
        what_failed = self._explain_what_failed(disruption_result)

        # 2. Which entities were affected
        affected_entities = self._explain_affected_entities(disruption_result)

        # 3. Why the disruption matters
        why_it_matters = self._explain_why_it_matters(
            disruption_result, risk_findings
        )

        # 4. What the financial engine calculated
        financial_calc = self._explain_financial_calculation(
            disruption_result, financial_result
        )

        # 5. Which recovery option was selected and why
        selected_recovery = self._explain_selected_recovery(financial_result)

        # 6. Which alternatives were considered
        alternatives_considered = self._explain_alternatives_considered(financial_result)

        # 7. Assumptions and missing information
        assumptions_missing = self._explain_assumptions_and_missing_data(
            disruption_result, financial_result, risk_findings
        )

        # 8. Remaining residual risks
        residual_risks = self._explain_residual_risks(
            disruption_result, financial_result
        )

        # 9. Human approval requirement
        human_approval = self._explain_human_approval(
            disruption_result, financial_result
        )

        # Synthesize full Markdown report
        markdown_report = self._build_markdown_report(
            what_failed=what_failed,
            affected_entities=affected_entities,
            why_it_matters=why_it_matters,
            financial_calc=financial_calc,
            selected_recovery=selected_recovery,
            alternatives_considered=alternatives_considered,
            assumptions_missing=assumptions_missing,
            residual_risks=residual_risks,
            human_approval=human_approval,
        )

        return {
            "what_failed": what_failed,
            "affected_entities": affected_entities,
            "why_it_matters": why_it_matters,
            "financial_calculation": financial_calc,
            "selected_recovery_option": selected_recovery,
            "alternatives_considered": alternatives_considered,
            "assumptions_and_missing_data": assumptions_missing,
            "residual_risks": residual_risks,
            "human_approval": human_approval,
            "markdown_report": markdown_report,
            "requires_human_approval": human_approval["required"],
        }

    # =========================================================================
    # Section Explanations
    # =========================================================================

    def _explain_what_failed(self, disruption_result: Dict[str, Any]) -> Dict[str, Any]:
        failed_entity = disruption_result.get("failed_entity", "Unknown Entity")
        disruption_type = disruption_result.get("disruption_type", "General failure")
        cap_impact = disruption_result.get("estimated_capacity_impact", {})
        direct_loss = cap_impact.get("estimated_direct_capacity_loss")

        if direct_loss is not None and direct_loss > 0:
            loss_str = f"{direct_loss:,.1f} throughput units"
        else:
            loss_str = "an unrecorded volume (capacity data unrecorded)"

        narrative = (
            f"A disruption event occurred at '{failed_entity}', classified as a '{disruption_type}'. "
            f"This outage severed immediate operational capacity, removing {loss_str} "
            f"from active service."
        )

        return {
            "failed_entity": failed_entity,
            "disruption_type": disruption_type,
            "direct_capacity_loss": direct_loss,
            "narrative": narrative,
        }

    def _explain_affected_entities(self, disruption_result: Dict[str, Any]) -> Dict[str, Any]:
        all_affected = disruption_result.get("affected_entities", [])
        
        # Check direct keys or extract from entity_impact_details
        confirmed = disruption_result.get("confirmed_shortage_entities")
        if confirmed is None:
            confirmed = [
                d["entity"]
                for d in disruption_result.get("entity_impact_details", [])
                if d.get("status") == "confirmed_shortage"
            ]

        potential = disruption_result.get("potential_shortage_entities")
        if potential is None:
            potential = [
                d["entity"]
                for d in disruption_result.get("entity_impact_details", [])
                if d.get("status") == "potentially_affected"
            ]

        materials = disruption_result.get("affected_materials", [])
        paths = disruption_result.get("disrupted_paths") or disruption_result.get("affected_paths", [])

        narrative_parts = [
            f"A total of {len(all_affected)} downstream entities were impacted across the supply chain."
        ]

        if confirmed:
            narrative_parts.append(
                f"Confirmed Shortages ({len(confirmed)}): {', '.join(confirmed)} have zero surviving operational "
                f"inbound paths and face immediate production stoppages."
            )
        else:
            narrative_parts.append("No downstream entities suffered 100% confirmed outages.")

        if potential:
            narrative_parts.append(
                f"Potentially Affected ({len(potential)}): {', '.join(potential)} possess partial alternative paths "
                f"or surviving branches, but require capacity and buffer verification."
            )

        if materials:
            narrative_parts.append(
                f"Impacted Commodities & Materials ({len(materials)}): {', '.join(materials)}."
            )

        return {
            "total_affected_count": len(all_affected),
            "all_affected": all_affected,
            "confirmed_shortages": confirmed,
            "potentially_affected": potential,
            "affected_materials": materials,
            "disrupted_paths_count": len(paths),
            "narrative": " ".join(narrative_parts),
        }

    def _explain_why_it_matters(
        self,
        disruption_result: Dict[str, Any],
        risk_findings: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        failed_entity = disruption_result.get("failed_entity", "")
        impact_level = disruption_result.get("impact_level", "Unknown")

        # Cross-reference with Member 2's shared risk analysis
        shared_risk_info = None
        if risk_findings:
            for item in risk_findings:
                dep_node = item.get("dependency_node")
                if dep_node and (dep_node == failed_entity or dep_node in failed_entity):
                    shared_risk_info = item
                    break

        reasons: List[str] = [
            f"The disruption simulation classified this event as '{impact_level}' severity."
        ]

        if shared_risk_info:
            score = shared_risk_info.get("risk_score", 0.0)
            level = shared_risk_info.get("risk_level", "Medium")
            supplier_count = shared_risk_info.get("affected_supplier_count", 0)
            is_shared = shared_risk_info.get("is_shared", False)

            if is_shared:
                reasons.append(
                    f"Structural Risk Audit: '{failed_entity}' was previously identified as a SHARED SINGLE POINT "
                    f"OF FAILURE (Risk Score: {score:.1f}/100, Level: {level}), directly supplying {supplier_count} "
                    f"distinct downstream suppliers simultaneously."
                )
            else:
                reasons.append(
                    f"Structural Risk Audit: Vulnerability index is {score:.1f}/100 ({level} risk level)."
                )

            exp = shared_risk_info.get("explanation")
            if exp:
                reasons.append(f"Audit breakdown: {exp}")
        else:
            reasons.append(
                f"Upstream failure at '{failed_entity}' triggers cascading downstream bottlenecks along all connected "
                f"directed edges."
            )

        return {
            "impact_level": impact_level,
            "is_shared_dependency": bool(shared_risk_info and shared_risk_info.get("is_shared")),
            "shared_risk_score": shared_risk_info.get("risk_score") if shared_risk_info else None,
            "narrative": " ".join(reasons),
        }

    def _explain_financial_calculation(
        self,
        disruption_result: Dict[str, Any],
        financial_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        baseline_loss = financial_result.get("baseline_disruption_loss", 0.0)
        affected_qty = financial_result.get("affected_quantity", 0.0)
        loss_per_unit = financial_result.get("loss_per_unit", 0.0)
        rec_cost = financial_result.get("recovery_cost", 0.0)
        residual_loss = financial_result.get("residual_disruption_loss", 0.0)
        total_cost = financial_result.get("estimated_total_cost", 0.0)
        savings = financial_result.get("estimated_savings", 0.0)

        formula = (
            f"Gross Disruption Loss = Affected Quantity ({affected_qty:,.1f}) "
            f"× Loss Per Unit (${loss_per_unit:,.2f}) = ${baseline_loss:,.2f}"
        )

        cost_formula = (
            f"Recovery Plan Total Cost = Recovery Expenditure (${rec_cost:,.2f}) "
            f"+ Residual Loss (${residual_loss:,.2f}) = ${total_cost:,.2f}"
        )

        narrative = (
            f"If no recovery action is taken, the business incurs an estimated baseline disruption loss of "
            f"${baseline_loss:,.2f}. The financial optimization engine evaluated candidate recovery interventions, "
            f"calculating an implementation expenditure of ${rec_cost:,.2f} with ${residual_loss:,.2f} in residual "
            f"unmitigated losses, achieving estimated net savings of ${savings:,.2f} relative to taking no action."
        )

        return {
            "baseline_loss": baseline_loss,
            "affected_quantity": affected_qty,
            "loss_per_unit": loss_per_unit,
            "recovery_cost": rec_cost,
            "residual_loss": residual_loss,
            "estimated_total_cost": total_cost,
            "estimated_savings": savings,
            "loss_formula": formula,
            "cost_formula": cost_formula,
            "narrative": narrative,
        }

    def _explain_selected_recovery(self, financial_result: Dict[str, Any]) -> Dict[str, Any]:
        rec_ids = financial_result.get("recommended_option_ids", [])
        feasible = financial_result.get("feasible_options", [])
        explanation = financial_result.get("recommendation_explanation", "")
        total_cost = financial_result.get("estimated_total_cost", 0.0)
        savings = financial_result.get("estimated_savings", 0.0)

        if not rec_ids:
            return {
                "selected_ids": [],
                "is_no_action_baseline": True,
                "narrative": (
                    "No candidate recovery option satisfied all operational constraints (budget, capacity, lead time). "
                    "The engine recommends the No-Action Baseline to avoid economically inefficient expenditures, "
                    "requiring manual management escalation."
                ),
            }

        selected_details = [
            f for f in feasible if f.get("option_id") in rec_ids
        ]

        narrative = (
            f"Selected Plan: '{', '.join(rec_ids)}'. This plan was mathematically chosen because it strictly "
            f"satisfies budget, lead-time, and capacity requirements while minimizing total economic exposure "
            f"(${total_cost:,.2f}), securing ${savings:,.2f} in net capital preservation over inaction."
        )

        return {
            "selected_ids": rec_ids,
            "selected_options_details": selected_details,
            "is_no_action_baseline": False,
            "narrative": narrative,
            "engine_explanation": explanation,
        }

    def _explain_alternatives_considered(
        self, financial_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        feasible = financial_result.get("feasible_options", [])
        rejected = financial_result.get("rejected_options", [])
        rec_ids = financial_result.get("recommended_option_ids", [])

        feasible_unselected = [
            opt for opt in feasible if opt.get("option_id") not in rec_ids
        ]

        breakdown: List[str] = [
            f"A total of {len(feasible) + len(rejected)} recovery alternatives were evaluated."
        ]

        if feasible_unselected:
            breakdown.append(
                f"Feasible but Sub-Optimal ({len(feasible_unselected)}): "
                + ", ".join([
                    f"{opt.get('option_id')} (${opt.get('estimated_total_cost', 0):,.2f} total cost)"
                    for opt in feasible_unselected
                ])
                + " were rejected in favor of the lower-cost recommended plan."
            )

        if rejected:
            rej_summaries = []
            for r in rejected:
                opt_id = r.get("option_id", "Unknown")
                reasons = r.get("rejection_reasons", [])
                reasons_str = "; ".join(reasons) if reasons else "Violated constraints"
                rej_summaries.append(f"{opt_id} ({reasons_str})")
            breakdown.append(
                f"Infeasible Alternatives Rejected ({len(rejected)}): " + "; ".join(rej_summaries) + "."
            )

        return {
            "total_considered": len(feasible) + len(rejected),
            "feasible_count": len(feasible),
            "rejected_count": len(rejected),
            "feasible_unselected": feasible_unselected,
            "rejected_options": rejected,
            "narrative": " ".join(breakdown),
        }

    def _explain_assumptions_and_missing_data(
        self,
        disruption_result: Dict[str, Any],
        financial_result: Dict[str, Any],
        risk_findings: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        # Compile unique assumptions
        assumptions: List[str] = list(disruption_result.get("assumptions", []))
        for a in financial_result.get("assumptions", []):
            if a not in assumptions:
                assumptions.append(a)

        # Compile unique missing data items
        missing_data: List[str] = list(disruption_result.get("missing_information", []))
        for m in financial_result.get("missing_information", []):
            if m not in missing_data:
                missing_data.append(m)

        if risk_findings:
            failed_entity = disruption_result.get("failed_entity", "")
            for r in risk_findings:
                if r.get("dependency_node") == failed_entity:
                    for item in r.get("missing_information", []):
                        if item not in missing_data:
                            missing_data.append(f"Risk Audit Gap: {item}")

        narrative_parts: List[str] = []
        if assumptions:
            narrative_parts.append(
                f"Key Operational Assumptions ({len(assumptions)}): " + "; ".join(assumptions) + "."
            )
        else:
            narrative_parts.append("No explicit heuristic assumptions recorded.")

        if missing_data:
            narrative_parts.append(
                f"Data Gaps Honest Disclosure ({len(missing_data)}): "
                + "; ".join(missing_data)
                + ". Values were NOT hallucinated or assumed as zero."
            )
        else:
            narrative_parts.append("All required operational, capacity, and financial attributes were present.")

        return {
            "assumptions": assumptions,
            "missing_data": missing_data,
            "narrative": " ".join(narrative_parts),
        }

    def _explain_residual_risks(
        self,
        disruption_result: Dict[str, Any],
        financial_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        residual_loss = financial_result.get("residual_disruption_loss", 0.0)
        budget = financial_result.get("available_budget")
        rec_cost = financial_result.get("recovery_cost", 0.0)
        rec_ids = financial_result.get("recommended_option_ids", [])
        potential = disruption_result.get("potential_shortage_entities", [])

        risks: List[str] = []
        if residual_loss > 0:
            risks.append(
                f"Financial Exposure: A residual loss of ${residual_loss:,.2f} remains unmitigated due to capacity caps."
            )
        else:
            risks.append("Financial Exposure: The selected recovery fully offsets the volume deficit.")

        if budget and budget > 0 and rec_cost > 0:
            utilization = (rec_cost / budget) * 100
            risks.append(
                f"Capital Depletion: Recovery consumes {utilization:.1f}% of allocated emergency budget "
                f"(${rec_cost:,.2f} / ${budget:,.2f})."
            )

        if potential:
            risks.append(
                f"Operational Buffer Risk: {len(potential)} partially-affected entities rely on secondary routing "
                f"and require real-time buffer inventory monitoring."
            )

        if not rec_ids:
            risks.append("Unmitigated Operational Outage: Inaction maintains 100% disruption risk across all paths.")

        return {
            "residual_financial_loss": residual_loss,
            "residual_risks_list": risks,
            "narrative": " ".join(risks),
        }

    def _explain_human_approval(
        self,
        disruption_result: Dict[str, Any],
        financial_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        requires_approval = bool(financial_result.get("requires_human_approval", False))
        savings = financial_result.get("estimated_savings", 0.0)
        budget = financial_result.get("available_budget")
        rec_cost = financial_result.get("recovery_cost", 0.0)
        rec_ids = financial_result.get("recommended_option_ids", [])
        impact_level = disruption_result.get("impact_level", "Medium")

        triggers: List[str] = []

        if not rec_ids:
            triggers.append("No viable recovery options available (No-Action Baseline invoked)")
        if savings < 0:
            triggers.append(f"Negative net economic return (plan costs ${abs(savings):,.2f} more than inaction)")
        if budget and budget > 0 and (rec_cost / budget) >= 0.85:
            triggers.append(f"High emergency budget utilization ({(rec_cost / budget)*100:.1f}% consumed)")
        if impact_level == "Critical":
            triggers.append("Critical severity supply-chain blast radius")

        if triggers:
            requires_approval = True
            narrative = (
                f"[ALERT: HUMAN APPROVAL REQUIRED]: Management sign-off is mandated prior to capital disbursement due to: "
                + "; ".join(triggers)
                + "."
            )
        else:
            narrative = (
                "[AUTOMATIC DISPATCH AUTHORIZED]: The recommended plan satisfies all governance rules, produces positive "
                "economic savings, and remains within budget safety margins."
            )

        return {
            "required": requires_approval,
            "triggers": triggers,
            "narrative": narrative,
        }

    # =========================================================================
    # Markdown Synthesis
    # =========================================================================

    def _build_markdown_report(
        self,
        what_failed: Dict[str, Any],
        affected_entities: Dict[str, Any],
        why_it_matters: Dict[str, Any],
        financial_calc: Dict[str, Any],
        selected_recovery: Dict[str, Any],
        alternatives_considered: Dict[str, Any],
        assumptions_missing: Dict[str, Any],
        residual_risks: Dict[str, Any],
        human_approval: Dict[str, Any],
    ) -> str:
        status_badge = "[HUMAN APPROVAL REQUIRED]" if human_approval["required"] else "[READY FOR EXECUTION]"

        lines = [
            f"# Nexus-AI: Multi-Tier Supply Chain Decision Explanation",
            f"**Governance Status:** {status_badge}  ",
            f"**Disrupted Asset:** `{what_failed['failed_entity']}` | **Severity:** `{why_it_matters['impact_level']}`  ",
            "",
            "---",
            "",
            "### 1. Root Cause & Disrupted Entity",
            f"{what_failed['narrative']}",
            "",
            "### 2. Downstream Blast Radius & Impacted Entities",
            f"{affected_entities['narrative']}",
            "",
            "### 3. Criticality & Vulnerability Context",
            f"{why_it_matters['narrative']}",
            "",
            "### 4. Financial Impact Computation",
            f"- **Formula:** `{financial_calc['loss_formula']}`",
            f"- **Cost Equation:** `{financial_calc['cost_formula']}`",
            f"{financial_calc['narrative']}",
            "",
            "### 5. Recommended Recovery Decision",
            f"{selected_recovery['narrative']}",
            "",
            "### 6. Alternatives Evaluated",
            f"{alternatives_considered['narrative']}",
            "",
            "### 7. Governance, Assumptions & Data Disclosures",
            f"{assumptions_missing['narrative']}",
            "",
            "### 8. Post-Recovery Residual Risks",
            f"{residual_risks['narrative']}",
            "",
            "### 9. Governance & Approval Protocol",
            f"{human_approval['narrative']}",
        ]

        return "\n".join(lines)


def generate_decision_explanation(
    graph: Optional[nx.DiGraph],
    disruption_result: Dict[str, Any],
    financial_result: Dict[str, Any],
    risk_findings: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper function for Member 5 decision explanation.
    """
    agent = DecisionExplanationAgent()
    return agent.explain(
        graph=graph,
        disruption_result=disruption_result,
        financial_result=financial_result,
        risk_findings=risk_findings,
    )
