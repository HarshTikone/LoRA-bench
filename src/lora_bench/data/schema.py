"""Record shapes used across the data-prep pipeline.

Two shapes matter here:

- The *raw* record: whatever `datasets.load_dataset("hitoshura25/cvefixes")`
  hands back per row. It's a plain dict; we don't wrap it in a class because
  it's untrusted/messy input (see cvefixes.clean_record) and we only read a
  fixed subset of its fields. RAW_FIELDS documents that subset, and
  cvefixes.load_raw_dataset validates the first row against it, so a schema
  change upstream raises immediately instead of silently dropping every row.
- FixDiffExample: the cleaned, instruction-tuning-ready record this
  pipeline produces, and what Day 2's fine-tuning step consumes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Fields this pipeline reads from a raw hitoshura25/cvefixes row. Confirmed
# against live rows on 2026-08-22 (see ADR-0002). Not exhaustive of the
# dataset's columns — only what clean_record() actually uses.
RAW_FIELDS = (
    "cve_id",
    "cve_description",
    "cwe_id",
    "cwe_name",
    "severity",
    "language",
    "repo_url",
    "hash",
    "vulnerable_code",
    "fixed_code",
    "diff_with_context",
)


@dataclass(frozen=True)
class FixDiffExample:
    """One cleaned CVE-fix instruction-tuning example."""

    cve_id: str
    cwe_id: str
    cwe_name: str
    severity: str  # normalized to "LOW" | "MEDIUM" | "HIGH" | "CRITICAL" | "UNKNOWN"
    language: str
    repo_url: str
    commit_hash: str

    instruction: str
    input: str  # vulnerable code
    output: str  # fixed code
    diff: str  # unified diff, kept for reporting/failure-case review, not training input

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FixDiffExample":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})
