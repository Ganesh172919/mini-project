from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ClaimInput(BaseModel):
    """Raw claim payload plus optional descriptive metadata.

    Only the eleven required intake fields below are transformed for the MLP.
    Optional identifiers/status are accepted as metadata and never enter the model.
    ClaimLegitimacy is intentionally absent because it is the training target.
    """

    model_config = ConfigDict(extra="forbid")

    ClaimAmount: float = Field(ge=0)
    ClaimDate: date
    PatientAge: int = Field(ge=0, le=120)
    PatientGender: Literal["F", "M"]
    ProviderSpecialty: Literal[
        "Cardiology",
        "General Practice",
        "Neurology",
        "Orthopedics",
        "Pediatrics",
    ]
    PatientIncome: float = Field(ge=0)
    PatientMaritalStatus: Literal["Divorced", "Married", "Single", "Widowed"]
    PatientEmploymentStatus: Literal["Employed", "Retired", "Student", "Unemployed"]
    ClaimType: Literal["Emergency", "Inpatient", "Outpatient", "Routine"]
    ClaimSubmissionMethod: Literal["Online", "Paper", "Phone"]
    Cluster: int = Field(ge=0, le=3)

    ClaimID: str | None = Field(default=None, description="Descriptive identifier; excluded from model input.")
    PatientID: str | None = Field(default=None, description="Descriptive identifier; excluded from model input.")
    ProviderID: str | None = Field(default=None, description="Descriptive identifier; excluded from model input.")
    DiagnosisCode: str | None = Field(default=None, description="High-cardinality code; excluded from model input.")
    ProcedureCode: str | None = Field(default=None, description="High-cardinality code; excluded from model input.")
    ClaimStatus: Literal["Approved", "Denied", "Pending"] | None = Field(
        default=None,
        description="Post-adjudication field; excluded from model input.",
    )
    ProviderLocation: str | None = Field(default=None, description="High-cardinality field; excluded from model input.")


class PredictionResponse(BaseModel):
    model_name: str
    model_version: str
    prediction: Literal["Fraud", "Legitimate"]
    fraud_score: float = Field(ge=0, le=1, description="Uncalibrated sigmoid model score, not a calibrated probability.")
    score_is_calibrated: bool
    feature_count: int
    feature_contract: str
    consumed_input_fields: list[str]
    excluded_input_fields: list[str]
