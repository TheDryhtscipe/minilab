"""Certificate library: the only code that can make a kill (design doc-gmj, "Falsifier")."""

from minilab.certify.check import CERTIFICATE, Certificate, check_certificate, check_kill
from minilab.certify.duality import DualityCertificate
from minilab.certify.exact import ExactCertificate
from minilab.certify.interval import IntervalCertificate
from minilab.certify.result import CheckResult

__all__ = [
    "CERTIFICATE",
    "Certificate",
    "CheckResult",
    "DualityCertificate",
    "ExactCertificate",
    "IntervalCertificate",
    "check_certificate",
    "check_kill",
]
