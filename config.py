"""Configuration constants and data models for the referral management system."""

from pydantic import BaseModel, Field
from typing import TypedDict

# Valid values for medical referral fields
VALID_PROCEDURE_TYPES = ['arthroplasty', 'soft_tissue', 'unknown']
VALID_BODY_PARTS = ['hip', 'knee', 'unknown']
VALID_ARTHROPLASTY_TYPES = ['primary', 'revision', 'na', 'unknown']
VALID_INJECTIONS = ['yes', 'no']
VALID_PHYSIOTHERAPY = ['yes', 'no']
VALID_FURTHER_INFO = ['yes', 'no']

class PatientInfo(TypedDict):
    name: str
    hospital_number: str

class MedicalReferral(BaseModel):
    procedure_type: str = Field(description="The type of procedure (arthroplasty, soft tissue)")
    body_part: str = Field(description="The body part involved (hip, knee)")
    arthroplasty_type: str = Field(description="The type of arthroplasty (primary, revision)")
    further_information_needed: str = Field(description="Whether further information is needed (yes, no)")
    had_injections: str = Field(description="Whether the patient had injections (yes, no, unknown)")
    had_physiotherapy: str = Field(description="Whether the patient had physiotherapy (yes, no, unknown)")

class AnalysisResult(TypedDict):
    procedure_type: str
    body_part: str
    arthroplasty_type: str
    further_information_needed: str
    had_injections: str
    had_physiotherapy: str
    confidence: float
    name: str
    hospital_number: str
    xray_findings: str