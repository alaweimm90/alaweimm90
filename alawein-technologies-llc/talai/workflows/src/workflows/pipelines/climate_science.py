"""
Climate Science Pipeline - Validation for climate models and predictions

Specialized pipeline for validating climate science workflows and predictions.
"""

import asyncio
from typing import Dict, Any, List, Optional
import logging
import statistics

from .validation_pipeline import ValidationPipeline, ValidationStep, ValidationResult, ValidationStatus


logger = logging.getLogger(__name__)


class ClimateSciencePipeline(ValidationPipeline):
    """Pipeline for climate science validation"""

    def __init__(self):
        super().__init__(
            name="climate_science",
            description="Validates climate models and predictions"
        )
        self._initialize_steps()

    def _initialize_steps(self):
        """Initialize climate science validation steps"""

        # Data source validation
        self.add_step(ValidationStep(
            name="data_sources",
            validator=self._validate_data_sources,
            required=True,
            timeout=30
        ))

        # Model calibration
        self.add_step(ValidationStep(
            name="model_calibration",
            validator=self._validate_calibration,
            required=True,
            timeout=60
        ))

        # Historical validation
        self.add_step(ValidationStep(
            name="historical_validation",
            validator=self._validate_historical,
            required=True,
            timeout=90
        ))

        # Uncertainty quantification
        self.add_step(ValidationStep(
            name="uncertainty_analysis",
            validator=self._validate_uncertainty,
            required=True,
            timeout=60
        ))

        # Physical consistency
        self.add_step(ValidationStep(
            name="physical_consistency",
            validator=self._validate_physics,
            required=False,
            timeout=45
        ))

        # Ensemble agreement
        self.add_step(ValidationStep(
            name="ensemble_agreement",
            validator=self._validate_ensemble,
            required=False,
            timeout=120
        ))

    async def _validate_data_sources(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate climate data sources"""
        sources = data.get("data_sources", [])
        temporal_coverage = data.get("temporal_coverage_years", 0)
        spatial_resolution = data.get("spatial_resolution_km", 100)

        if not sources:
            return {
                "status": "failed",
                "message": "No data sources specified"
            }

        # Check temporal coverage
        if temporal_coverage < 30:
            return {
                "status": "warning",
                "score": temporal_coverage / 30,
                "message": f"Limited temporal coverage: {temporal_coverage} years"
            }

        # Check spatial resolution
        resolution_score = 1.0
        if spatial_resolution > 50:
            resolution_score = 50 / spatial_resolution

        # Check source reliability
        reliable_sources = ["NASA", "NOAA", "ECMWF", "ERA5", "CMIP6"]
        source_score = sum(1 for s in sources if any(r in s.upper() for r in reliable_sources)) / len(sources)

        overall_score = (source_score + resolution_score) / 2

        await asyncio.sleep(0.5)

        return {
            "status": "passed" if overall_score > 0.7 else "warning",
            "score": overall_score,
            "message": f"Using {len(sources)} data sources",
            "details": {
                "sources": sources,
                "temporal_coverage": temporal_coverage,
                "spatial_resolution": spatial_resolution
            }
        }

    async def _validate_calibration(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model calibration"""
        calibration_r2 = data.get("calibration_r2", 0)
        rmse = data.get("rmse", float('inf'))
        bias = data.get("bias", 0)

        # Check R-squared
        if calibration_r2 < 0.7:
            return {
                "status": "failed",
                "score": calibration_r2,
                "message": f"Poor calibration: R² = {calibration_r2:.3f}"
            }

        # Check bias
        if abs(bias) > 2:  # degrees Celsius
            status = "warning"
            message = f"Significant bias: {bias:.1f}°C"
            score = calibration_r2 * 0.8
        else:
            status = "passed"
            message = "Model well calibrated"
            score = calibration_r2

        await asyncio.sleep(0.5)

        return {
            "status": status,
            "score": score,
            "message": message,
            "details": {
                "r_squared": calibration_r2,
                "rmse": rmse,
                "bias": bias
            }
        }

    async def _validate_historical(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate against historical data"""
        hindcast_skill = data.get("hindcast_skill", 0)
        trend_accuracy = data.get("trend_accuracy", 0)
        extreme_events_captured = data.get("extreme_events_captured", 0)

        # Calculate overall historical validation score
        scores = [hindcast_skill, trend_accuracy, extreme_events_captured]
        avg_score = statistics.mean(scores)

        if avg_score < 0.6:
            status = "failed"
            message = "Poor historical performance"
        elif avg_score < 0.8:
            status = "warning"
            message = "Moderate historical performance"
        else:
            status = "passed"
            message = "Good historical validation"

        await asyncio.sleep(1.0)

        return {
            "status": status,
            "score": avg_score,
            "message": message,
            "details": {
                "hindcast_skill": hindcast_skill,
                "trend_accuracy": trend_accuracy,
                "extreme_events_captured": extreme_events_captured
            }
        }

    async def _validate_uncertainty(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate uncertainty quantification"""
        uncertainty_bounds = data.get("uncertainty_bounds", {})
        confidence_level = data.get("confidence_level", 0.95)
        sources_of_uncertainty = data.get("uncertainty_sources", [])

        if not uncertainty_bounds:
            return {
                "status": "failed",
                "message": "No uncertainty quantification provided"
            }

        # Check if uncertainty is reasonable
        lower_bound = uncertainty_bounds.get("lower", 0)
        upper_bound = uncertainty_bounds.get("upper", 0)
        uncertainty_range = upper_bound - lower_bound

        if uncertainty_range > 10:  # degrees Celsius
            status = "warning"
            score = 0.5
            message = f"Large uncertainty range: {uncertainty_range:.1f}°C"
        else:
            status = "passed"
            score = 1 - uncertainty_range / 10
            message = "Uncertainty well quantified"

        await asyncio.sleep(0.5)

        return {
            "status": status,
            "score": score,
            "message": message,
            "details": {
                "uncertainty_range": uncertainty_range,
                "confidence_level": confidence_level,
                "sources": sources_of_uncertainty
            }
        }

    async def _validate_physics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate physical consistency"""
        energy_balance = data.get("energy_balance_error", 0)
        mass_conservation = data.get("mass_conservation_error", 0)
        thermodynamic_consistency = data.get("thermodynamic_consistency", True)

        violations = []

        # Check energy balance
        if abs(energy_balance) > 0.01:
            violations.append(f"Energy imbalance: {energy_balance:.3f} W/m²")

        # Check mass conservation
        if abs(mass_conservation) > 0.001:
            violations.append(f"Mass conservation error: {mass_conservation:.4f}")

        # Check thermodynamics
        if not thermodynamic_consistency:
            violations.append("Thermodynamic inconsistency detected")

        await asyncio.sleep(0.5)

        if violations:
            return {
                "status": "warning",
                "score": 1 - len(violations) * 0.3,
                "message": f"Physical violations: {'; '.join(violations)}",
                "details": {"violations": violations}
            }

        return {
            "status": "passed",
            "score": 0.95,
            "message": "Physically consistent",
            "details": {
                "energy_balance": energy_balance,
                "mass_conservation": mass_conservation
            }
        }

    async def _validate_ensemble(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate ensemble model agreement"""
        ensemble_members = data.get("ensemble_members", [])
        agreement_metric = data.get("ensemble_agreement", 0)

        if not ensemble_members:
            return {
                "status": "skipped",
                "message": "No ensemble data provided"
            }

        if len(ensemble_members) < 3:
            return {
                "status": "warning",
                "score": 0.5,
                "message": f"Small ensemble size: {len(ensemble_members)} members"
            }

        # Check ensemble spread
        if ensemble_members:
            ensemble_std = statistics.stdev(ensemble_members)
            ensemble_mean = statistics.mean(ensemble_members)
            cv = ensemble_std / abs(ensemble_mean) if ensemble_mean != 0 else float('inf')
        else:
            cv = 0

        if cv > 0.3:
            status = "warning"
            score = 1 - cv
            message = "High ensemble spread"
        else:
            status = "passed"
            score = agreement_metric
            message = f"Good ensemble agreement ({len(ensemble_members)} members)"

        await asyncio.sleep(0.5)

        return {
            "status": status,
            "score": score,
            "message": message,
            "details": {
                "ensemble_size": len(ensemble_members),
                "coefficient_variation": cv,
                "agreement_metric": agreement_metric
            }
        }