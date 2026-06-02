from __future__ import annotations

from pydantic import BaseModel, Field


class RiskBucket(BaseModel):
    lower: float = Field(ge=0.0, le=1.0)
    upper: float = Field(ge=0.0, le=1.0)
    count: int = Field(ge=0)
    mean_predicted_risk: float = Field(ge=0.0, le=1.0)
    empirical_miss_frequency: float = Field(ge=0.0, le=1.0)


class MissRiskReport(BaseModel):
    split: str
    brier_score: float | None = None
    expected_calibration_error: float | None = None
    buckets: list[RiskBucket] = Field(default_factory=list)
