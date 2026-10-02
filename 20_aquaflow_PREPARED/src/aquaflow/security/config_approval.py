"""Multi-party countersign approval for sensitive runtime configuration changes.

Every change declares its impact scope, applicant, risk level and the number of
approvers required. Approvals are independent, timestamped records bound to one
specific change version and are never reused across versions. A change becomes
effective only after the countersign quorum is met AND all validators pass.
Rejections and withdrawals are recorded in an append-only audit trail.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Callable, Iterable

RISK_LEVELS = ("low", "medium", "high", "critical")
MIN_REQUIRED_APPROVALS = 2  # countersign requires at least two distinct approvers

PENDING = "pending"
APPROVED = "approved"
EFFECTIVE = "effective"
REJECTED = "rejected"
WITHDRAWN = "withdrawn"
VALIDATION_FAILED = "validation_failed"

DECIDABLE_STATES = (PENDING, APPROVED)
CLOSED_STATES = (EFFECTIVE, WITHDRAWN)

Validator = Callable[["ChangeRequest"], None]

_UNSET = object()


@dataclass(frozen=True)
class ChangeRequest:
    """One version of a proposed sensitive runtime configuration change."""

    change_id: str
    version: int
    config_key: str
    proposed_value: Any
    impact_scope: tuple[str, ...]
    applicant: str
    risk_level: str
    required_approvals: int
    state: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class ApprovalRecord:
    """One independent, timestamped decision bound to exactly one change version."""

    record_id: str
    change_id: str
    version: int
    approver: str
    decision: str  # "approve" | "reject"
    reason: str
    at: datetime


@dataclass(frozen=True)
class AuditEvent:
    """One append-only audit trail entry; rejections and withdrawals included."""

    seq: int
    at: datetime
    actor: str
    action: str
    change_id: str
    version: int
    detail: str


def _normalize_scope(impact_scope: Iterable[str]) -> tuple[str, ...]:
    if isinstance(impact_scope, str):
        raise ValueError("impact_scope must be a collection of scope names")
    scope = tuple(str(item).strip() for item in impact_scope)
    if not scope or any(not item for item in scope):
        raise ValueError("impact_scope must declare at least one affected scope")
    return scope


def _validate_declaration(config_key: str, applicant: str, risk_level: str, required_approvals: int) -> None:
    if not config_key or not isinstance(config_key, str):
        raise ValueError("config_key required")
    if not applicant or not isinstance(applicant, str):
        raise ValueError("applicant required")
    if risk_level not in RISK_LEVELS:
        raise ValueError(f"risk_level must be one of {RISK_LEVELS}")
    if isinstance(required_approvals, bool) or not isinstance(required_approvals, int):
        raise ValueError("required_approvals must be an integer")
    if required_approvals < MIN_REQUIRED_APPROVALS:
        raise ValueError(f"countersign requires at least {MIN_REQUIRED_APPROVALS} approvers")


class ConfigChangeApprovalService:
    """Versioned countersign workflow over a sensitive runtime config store."""

    def __init__(self, initial_config: dict[str, Any] | None = None):
        self._config = dict(initial_config or {})
        self._requests: dict[str, ChangeRequest] = {}
        self._decisions: dict[str, list[ApprovalRecord]] = {}
        self._audit: list[AuditEvent] = []
        self._validators: dict[str, list[Validator]] = {}
        self._seq = 0

    # -- queries ---------------------------------------------------------
    def get(self, change_id: str) -> ChangeRequest:
        return self._require(change_id)

    def current_config(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def decisions(self, change_id: str, version: int | None = None) -> list[ApprovalRecord]:
        self._require(change_id)
        records = self._decisions[change_id]
        if version is None:
            return list(records)
        return [record for record in records if record.version == version]

    def approvals(self, change_id: str, version: int | None = None) -> list[ApprovalRecord]:
        request = self._require(change_id)
        target = request.version if version is None else version
        return [r for r in self._decisions[change_id] if r.version == target and r.decision == "approve"]

    def audit_trail(self, change_id: str | None = None) -> list[AuditEvent]:
        if change_id is None:
            return list(self._audit)
        return [event for event in self._audit if event.change_id == change_id]

    def register_validator(self, config_key: str, validator: Validator) -> None:
        """Attach an activation-time check; validator raises ValueError on failure."""
        self._validators.setdefault(config_key, []).append(validator)

    # -- lifecycle -------------------------------------------------------
    def submit(
        self,
        change_id: str,
        config_key: str,
        proposed_value: Any,
        impact_scope: Iterable[str],
        applicant: str,
        risk_level: str,
        required_approvals: int,
        at: datetime,
    ) -> ChangeRequest:
        if change_id in self._requests:
            raise ValueError(f"duplicate change_id {change_id!r}")
        scope = _normalize_scope(impact_scope)
        _validate_declaration(config_key, applicant, risk_level, required_approvals)
        request = ChangeRequest(
            change_id, 1, config_key, proposed_value, scope,
            applicant, risk_level, required_approvals, PENDING, at, at,
        )
        self._requests[change_id] = request
        self._decisions[change_id] = []
        self._log(at, applicant, "submit", change_id, 1,
                  f"config_key={config_key} risk={risk_level} required={required_approvals} scope={','.join(scope)}")
        return request

    def approve(self, change_id: str, version: int, approver: str, at: datetime, reason: str = "") -> ApprovalRecord:
        request, record = self._record_decision(change_id, version, approver, "approve", reason, at)
        self._log(at, approver, "approve", change_id, version, f"record={record.record_id}")
        approvals = [r for r in self._decisions[change_id] if r.version == version and r.decision == "approve"]
        if request.state == PENDING and len(approvals) >= request.required_approvals:
            self._requests[change_id] = replace(request, state=APPROVED, updated_at=at)
            self._log(at, approver, "quorum_met", change_id, version,
                      f"approvals={len(approvals)}/{request.required_approvals}")
        else:
            self._requests[change_id] = replace(request, updated_at=at)
        return record

    def reject(self, change_id: str, version: int, approver: str, at: datetime, reason: str) -> ApprovalRecord:
        if not reason:
            raise ValueError("rejection reason required")
        request, record = self._record_decision(change_id, version, approver, "reject", reason, at)
        self._requests[change_id] = replace(request, state=REJECTED, updated_at=at)
        self._log(at, approver, "reject", change_id, version, f"record={record.record_id} reason={reason}")
        return record

    def withdraw(self, change_id: str, actor: str, at: datetime, reason: str) -> ChangeRequest:
        if not reason:
            raise ValueError("withdrawal reason required")
        request = self._require(change_id)
        if actor != request.applicant:
            raise ValueError("only the applicant can withdraw a change")
        if request.state not in DECIDABLE_STATES:
            raise ValueError(f"cannot withdraw in state {request.state!r}")
        self._check_clock(request, at)
        request = replace(request, state=WITHDRAWN, updated_at=at)
        self._requests[change_id] = request
        self._log(at, actor, "withdraw", change_id, request.version, f"reason={reason}")
        return request

    def activate(self, change_id: str, version: int, at: datetime, actor: str = "system") -> ChangeRequest:
        """Apply the change only if quorum is met and every validator passes."""
        request = self._require(change_id)
        if version != request.version:
            raise ValueError(f"stale version {version}; current version is {request.version}")
        if request.state == PENDING:
            raise ValueError("countersign quorum not met")
        if request.state != APPROVED:
            raise ValueError(f"cannot activate in state {request.state!r}")
        self._check_clock(request, at)
        errors = []
        for validator in self._validators.get(request.config_key, []):
            try:
                validator(request)
            except ValueError as exc:
                errors.append(str(exc))
        if errors:
            request = replace(request, state=VALIDATION_FAILED, updated_at=at)
            self._requests[change_id] = request
            self._log(at, actor, "validation_failed", change_id, version, "; ".join(errors))
            return request
        self._config[request.config_key] = request.proposed_value
        request = replace(request, state=EFFECTIVE, updated_at=at)
        self._requests[change_id] = request
        self._log(at, actor, "effective", change_id, version, f"config_key={request.config_key}")
        return request

    def revise(
        self,
        change_id: str,
        actor: str,
        at: datetime,
        *,
        proposed_value: Any = _UNSET,
        impact_scope: Iterable[str] | None = None,
        risk_level: str | None = None,
        required_approvals: int | None = None,
    ) -> ChangeRequest:
        """Open the next version; prior approvals stay bound to the old version."""
        request = self._require(change_id)
        if actor != request.applicant:
            raise ValueError("only the applicant can revise a change")
        if request.state in CLOSED_STATES:
            raise ValueError(f"cannot revise in state {request.state!r}")
        self._check_clock(request, at)
        new_value = request.proposed_value if proposed_value is _UNSET else proposed_value
        new_scope = request.impact_scope if impact_scope is None else _normalize_scope(impact_scope)
        new_risk = request.risk_level if risk_level is None else risk_level
        new_required = request.required_approvals if required_approvals is None else required_approvals
        _validate_declaration(request.config_key, request.applicant, new_risk, new_required)
        revised = ChangeRequest(
            change_id, request.version + 1, request.config_key, new_value, new_scope,
            request.applicant, new_risk, new_required, PENDING, request.created_at, at,
        )
        self._requests[change_id] = revised
        self._log(at, actor, "revise", change_id, revised.version,
                  f"supersedes v{request.version}; approvals reset")
        return revised

    # -- internals -------------------------------------------------------
    def _require(self, change_id: str) -> ChangeRequest:
        try:
            return self._requests[change_id]
        except KeyError:
            raise KeyError(f"unknown change {change_id!r}") from None

    def _log(self, at: datetime, actor: str, action: str, change_id: str, version: int, detail: str) -> None:
        self._seq += 1
        self._audit.append(AuditEvent(self._seq, at, actor, action, change_id, version, detail))

    @staticmethod
    def _check_clock(request: ChangeRequest, at: datetime) -> None:
        if at < request.updated_at:
            raise ValueError("time reversal")

    def _record_decision(
        self,
        change_id: str,
        version: int,
        approver: str,
        decision: str,
        reason: str,
        at: datetime,
    ) -> tuple[ChangeRequest, ApprovalRecord]:
        if not approver:
            raise ValueError("approver required")
        request = self._require(change_id)
        if request.state not in DECIDABLE_STATES:
            raise ValueError(f"cannot {decision} in state {request.state!r}")
        if version != request.version:
            raise ValueError(f"stale version {version}; current version is {request.version}")
        if approver == request.applicant:
            raise ValueError("applicant cannot decide on their own change")
        if any(r.version == version and r.approver == approver for r in self._decisions[change_id]):
            raise ValueError(f"{approver!r} already decided on {change_id} v{version}")
        self._check_clock(request, at)
        record = ApprovalRecord(
            record_id=f"{change_id}/v{version}/d{len(self._decisions[change_id]) + 1}",
            change_id=change_id,
            version=version,
            approver=approver,
            decision=decision,
            reason=reason,
            at=at,
        )
        self._decisions[change_id].append(record)
        return request, record
