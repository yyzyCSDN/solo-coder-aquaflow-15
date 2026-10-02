"""Multi-party countersignature approval for sensitive runtime config changes.

A change to a sensitive runtime configuration takes effect only when:

1. its declaration states the impact *scope*, the *applicant*, a *risk level*
   and the *required number of approvals*;
2. that many distinct approvers have independently countersigned the *same*
   version; and
3. every finalization check passes (segregation of duties, optimistic
   concurrency against the live value and the registered config validators).

Approval records are immutable, independently timestamped and always bound to
exactly one change version - they are never carried over to a revision.
Rejections and withdrawals are kept as append-only audit events, and a new
(superseding) version always starts with zero approvals.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Callable

RISK_LOW = "low"
RISK_MEDIUM = "medium"
RISK_HIGH = "high"
RISK_CRITICAL = "critical"
RISK_LEVELS = (RISK_LOW, RISK_MEDIUM, RISK_HIGH, RISK_CRITICAL)

# High-impact changes can never be approved by a single person.
MIN_APPROVALS_BY_RISK = {
    RISK_LOW: 1,
    RISK_MEDIUM: 1,
    RISK_HIGH: 2,
    RISK_CRITICAL: 2,
}

STATUS_PENDING = "pending"
STATUS_EFFECTIVE = "effective"
STATUS_REJECTED = "rejected"
STATUS_WITHDRAWN = "withdrawn"
STATUS_SUPERSEDED = "superseded"
STATUS_FAILED = "failed"
TERMINAL_STATUSES = (
    STATUS_EFFECTIVE,
    STATUS_REJECTED,
    STATUS_WITHDRAWN,
    STATUS_SUPERSEDED,
    STATUS_FAILED,
)
REVISABLE_STATUSES = (
    STATUS_EFFECTIVE,
    STATUS_REJECTED,
    STATUS_WITHDRAWN,
    STATUS_FAILED,
)

DECISION_APPROVED = "approved"
DECISION_REJECTED = "rejected"

# A validator receives (key, old_value, new_value, live_config) and returns
# an error message when the new value is unacceptable, otherwise None.
Validator = Callable[[str, object, object, "dict[str, object]"], "str | None"]

_UNSET = object()


class ChangeError(ValueError):
    """Base class for change-approval failures."""


class ChangeNotFound(ChangeError):
    """The change id (or one of its versions) does not exist."""


class InvalidDeclaration(ChangeError):
    """The change declaration is incomplete or violates policy."""


class InvalidState(ChangeError):
    """The action is not allowed in the change's current status."""


class DuplicateApproval(ChangeError):
    """The approver has already countersigned this exact version."""


class UnauthorizedApprover(ChangeError):
    """The actor may not perform that approval action on this change."""


class TimeOrderViolation(ChangeError):
    """A timestamp is naive or not after the previous recorded event."""


@dataclass(frozen=True)
class ChangeRequest:
    change_id: str
    version: int
    config_key: str
    old_value: object
    new_value: object
    scope: str
    applicant: str
    risk: str
    required_approvals: int
    status: str
    created_at: datetime
    reason: str = ""
    effective_at: "datetime | None" = None


@dataclass(frozen=True)
class ApprovalRecord:
    """An independent, timestamped decision bound to one change version."""

    record_id: str
    change_id: str
    version: int
    approver: str
    decision: str
    at: datetime
    note: str = ""


@dataclass(frozen=True)
class AuditEvent:
    event_id: str
    change_id: str
    version: int
    kind: str
    actor: str
    at: datetime
    detail: str = ""


@dataclass(frozen=True)
class ApprovalResult:
    change_id: str
    version: int
    approvals: int
    required_approvals: int
    quorum_reached: bool
    effective: bool
    status: str
    errors: "tuple[str, ...]" = field(default_factory=tuple)


class ChangeApprovalService:
    """In-memory, append-only store of sensitive config change approvals."""

    def __init__(self) -> None:
        self._config: "dict[str, object]" = {}
        self._validators: "dict[str, list[Validator]]" = {}
        self._versions: "dict[str, list[ChangeRequest]]" = {}
        self._records: "dict[str, list[ApprovalRecord]]" = {}
        self._events: "dict[str, list[AuditEvent]]" = {}
        self._last_event_at: "dict[str, datetime]" = {}
        self._change_seq = 0
        self._record_seq = 0
        self._event_seq = 0

    # -- configuration registry ------------------------------------------------

    def register_config(
        self,
        key: str,
        initial: object = None,
        validators: "list[Validator] | None" = None,
    ) -> str:
        if not isinstance(key, str) or not key.strip():
            raise InvalidDeclaration("config key must be a non-empty string")
        if key in self._config:
            raise InvalidDeclaration(f"config {key!r} is already registered")
        self._config[key] = initial
        self._validators[key] = list(validators or ())
        return key

    def effective_value(self, key: str) -> object:
        if key not in self._config:
            raise ChangeNotFound(f"unknown config {key!r}")
        return self._config[key]

    # -- submission -------------------------------------------------------------

    def submit(
        self,
        *,
        config_key: str,
        new_value: object,
        scope: str,
        applicant: str,
        risk: str,
        required_approvals: int,
        at: datetime,
        reason: str = "",
        old_value: object = _UNSET,
    ) -> ChangeRequest:
        self._require_aware(at)
        if config_key not in self._config:
            raise InvalidDeclaration(f"unknown sensitive config {config_key!r}")
        self._validate_declaration(scope, applicant, risk, required_approvals)
        if old_value is _UNSET:
            old_value = self._config[config_key]
        if old_value == new_value:
            raise InvalidDeclaration("new_value must differ from old_value")

        self._change_seq += 1
        change_id = f"CHG-{self._change_seq:04d}"
        request = ChangeRequest(
            change_id=change_id,
            version=1,
            config_key=config_key,
            old_value=old_value,
            new_value=new_value,
            scope=scope.strip(),
            applicant=applicant.strip(),
            risk=risk,
            required_approvals=required_approvals,
            status=STATUS_PENDING,
            created_at=at,
            reason=reason,
        )
        self._versions[change_id] = [request]
        self._records[change_id] = []
        self._events[change_id] = []
        self._last_event_at[change_id] = at
        self._emit(request, "created", request.applicant, at, reason)
        return request

    def revise(
        self,
        change_id: str,
        at: datetime,
        *,
        new_value: object = _UNSET,
        scope: "str | None" = None,
        risk: "str | None" = None,
        required_approvals: "int | None" = None,
        reason: str = "",
    ) -> ChangeRequest:
        """Open a new version; no approval records carry over from the old one."""
        current = self.get(change_id)
        self._require_aware(at)
        self._check_time(change_id, at)
        if current.status not in REVISABLE_STATUSES:
            raise InvalidState(
                f"cannot revise a {current.status!r} change; withdraw or resolve it first"
            )
        if current.config_key not in self._config:
            raise InvalidDeclaration(
                f"config {current.config_key!r} is no longer registered"
            )

        new_value = current.new_value if new_value is _UNSET else new_value
        scope = current.scope if scope is None else scope
        risk = current.risk if risk is None else risk
        required_approvals = (
            current.required_approvals
            if required_approvals is None
            else required_approvals
        )
        self._validate_declaration(scope, current.applicant, risk, required_approvals)

        # A revision is based on the value that is live now, not on the stale
        # value the previous version was proposed against.
        old_value = self._config[current.config_key]
        if old_value == new_value:
            raise InvalidDeclaration("new_value must differ from the live value")

        superseded = replace(current, status=STATUS_SUPERSEDED)
        self._store(superseded)
        self._emit(superseded, "superseded", current.applicant, at, reason)

        revision = ChangeRequest(
            change_id=change_id,
            version=current.version + 1,
            config_key=current.config_key,
            old_value=old_value,
            new_value=new_value,
            scope=scope.strip(),
            applicant=current.applicant,
            risk=risk,
            required_approvals=required_approvals,
            status=STATUS_PENDING,
            created_at=at,
            reason=reason,
        )
        self._versions[change_id].append(revision)
        self._emit(revision, "created", revision.applicant, at, reason)
        return revision

    # -- approval actions -------------------------------------------------------

    def approve(
        self, change_id: str, approver: str, at: datetime, note: str = ""
    ) -> ApprovalResult:
        request = self._require_pending(change_id, at)
        approver = self._require_actor(approver)
        if approver == request.applicant:
            raise UnauthorizedApprover(
                "the applicant cannot countersign their own change"
            )
        already = self._version_records(
            change_id, request.version, DECISION_APPROVED
        )
        if any(record.approver == approver for record in already):
            raise DuplicateApproval(
                f"{approver!r} has already approved {change_id} v{request.version}"
            )

        record = self._append_record(request, approver, DECISION_APPROVED, at, note)
        self._emit(request, "approved", approver, at, note)

        approvals = len(
            self._version_records(change_id, request.version, DECISION_APPROVED)
        )
        if approvals < request.required_approvals:
            return ApprovalResult(
                change_id=change_id,
                version=request.version,
                approvals=approvals,
                required_approvals=request.required_approvals,
                quorum_reached=False,
                effective=False,
                status=STATUS_PENDING,
            )
        return self._finalize(request, at)

    def reject(
        self, change_id: str, approver: str, at: datetime, reason: str
    ) -> ChangeRequest:
        if not isinstance(reason, str) or not reason.strip():
            raise InvalidDeclaration("a rejection reason is required for audit")
        request = self._require_pending(change_id, at)
        approver = self._require_actor(approver)
        if approver == request.applicant:
            raise UnauthorizedApprover(
                "the applicant rejects by withdrawing, not by rejecting"
            )

        self._append_record(request, approver, DECISION_REJECTED, at, reason)
        rejected = replace(request, status=STATUS_REJECTED)
        self._store(rejected)
        self._emit(rejected, "rejected", approver, at, reason)
        return rejected

    def withdraw(
        self,
        change_id: str,
        at: datetime,
        reason: str,
        actor: "str | None" = None,
    ) -> ChangeRequest:
        if not isinstance(reason, str) or not reason.strip():
            raise InvalidDeclaration("a withdrawal reason is required for audit")
        request = self._require_pending(change_id, at)
        actor = request.applicant if actor is None else self._require_actor(actor)
        if actor != request.applicant:
            raise UnauthorizedApprover("only the applicant may withdraw the change")

        withdrawn = replace(request, status=STATUS_WITHDRAWN)
        self._store(withdrawn)
        self._emit(withdrawn, "withdrawn", actor, at, reason)
        return withdrawn

    # -- read side ---------------------------------------------------------------

    def get(self, change_id: str, version: "int | None" = None) -> ChangeRequest:
        if change_id not in self._versions:
            raise ChangeNotFound(f"unknown change {change_id!r}")
        if version is None:
            return self._versions[change_id][-1]
        versions = self._versions[change_id]
        if not 1 <= version <= len(versions):
            raise ChangeNotFound(f"{change_id} has no version {version}")
        return versions[version - 1]

    def approvals(
        self, change_id: str, version: "int | None" = None
    ) -> "tuple[ApprovalRecord, ...]":
        """Independent countersign records for one version (default: current).

        Only records with an ``approved`` decision are returned; rejections
        remain available via :meth:`decisions`.
        """
        target_version = self.get(change_id, version).version
        return tuple(
            record
            for record in self._records[change_id]
            if record.version == target_version
            and record.decision == DECISION_APPROVED
        )

    def decisions(
        self, change_id: str, version: "int | None" = None
    ) -> "tuple[ApprovalRecord, ...]":
        """All independent decision records (approvals and rejections)."""
        target_version = self.get(change_id, version).version
        return tuple(
            record
            for record in self._records[change_id]
            if record.version == target_version
        )

    def history(self, change_id: str) -> "tuple[AuditEvent, ...]":
        if change_id not in self._events:
            raise ChangeNotFound(f"unknown change {change_id!r}")
        return tuple(self._events[change_id])

    def approvals_remaining(self, change_id: str) -> int:
        request = self.get(change_id)
        have = len(
            self._version_records(
                change_id, request.version, DECISION_APPROVED
            )
        )
        return max(0, request.required_approvals - have)

    # -- internals ---------------------------------------------------------------

    def _finalize(self, request: ChangeRequest, at: datetime) -> ApprovalResult:
        """Run every post-quorum check; the change is applied only if all pass."""
        errors: "list[str]" = []
        records = self._version_records(
            request.change_id, request.version, DECISION_APPROVED
        )
        approvers = {record.approver for record in records}
        if len(approvers) != len(records):
            errors.append("duplicate approver detected on this version")
        if len(approvers) < request.required_approvals:
            errors.append(
                f"quorum not reached: {len(approvers)}/{request.required_approvals}"
            )
        required_by_risk = MIN_APPROVALS_BY_RISK[request.risk]
        if len(approvers) < required_by_risk:
            errors.append(
                f"risk {request.risk!r} requires at least {required_by_risk} approvals"
            )
        if request.applicant in approvers:
            errors.append("the applicant may not be among the approvers")

        if request.config_key not in self._config:
            errors.append(f"config {request.config_key!r} is no longer registered")
        else:
            if request.old_value != self._config[request.config_key]:
                errors.append(
                    "the live value changed before enactment; this version is stale"
                )
            for validator in self._validators[request.config_key]:
                message = validator(
                    request.config_key,
                    request.old_value,
                    request.new_value,
                    dict(self._config),
                )
                if message:
                    errors.append(str(message))

        if errors:
            failed = replace(request, status=STATUS_FAILED)
            self._store(failed)
            self._emit(failed, "failed", "system", at, "; ".join(errors))
            return ApprovalResult(
                change_id=request.change_id,
                version=request.version,
                approvals=len(approvers),
                required_approvals=request.required_approvals,
                quorum_reached=len(approvers) >= request.required_approvals,
                effective=False,
                status=STATUS_FAILED,
                errors=tuple(errors),
            )

        self._config[request.config_key] = request.new_value
        effective = replace(request, status=STATUS_EFFECTIVE, effective_at=at)
        self._store(effective)
        self._emit(effective, "effective", "system", at)
        return ApprovalResult(
            change_id=request.change_id,
            version=request.version,
            approvals=len(approvers),
            required_approvals=request.required_approvals,
            quorum_reached=True,
            effective=True,
            status=STATUS_EFFECTIVE,
        )

    def _validate_declaration(
        self,
        scope: str,
        applicant: str,
        risk: str,
        required_approvals: int,
    ) -> None:
        if not isinstance(scope, str) or not scope.strip():
            raise InvalidDeclaration("impact scope must be declared")
        if not isinstance(applicant, str) or not applicant.strip():
            raise InvalidDeclaration("applicant must be declared")
        if risk not in RISK_LEVELS:
            raise InvalidDeclaration(f"unknown risk level {risk!r}")
        if not isinstance(required_approvals, int) or required_approvals < 1:
            raise InvalidDeclaration("required_approvals must be a positive integer")
        minimum = MIN_APPROVALS_BY_RISK[risk]
        if required_approvals < minimum:
            raise InvalidDeclaration(
                f"risk {risk!r} requires at least {minimum} approvals"
            )

    def _require_pending(self, change_id: str, at: datetime) -> ChangeRequest:
        request = self.get(change_id)
        self._require_aware(at)
        self._check_time(change_id, at)
        if request.status != STATUS_PENDING:
            raise InvalidState(
                f"{change_id} v{request.version} is {request.status!r}, not pending"
            )
        return request

    def _version_records(
        self, change_id: str, version: int, decision: "str | None" = None
    ) -> "list[ApprovalRecord]":
        records = [
            record
            for record in self._records[change_id]
            if record.version == version
        ]
        if decision is not None:
            records = [record for record in records if record.decision == decision]
        return records

    def _append_record(
        self,
        request: ChangeRequest,
        approver: str,
        decision: str,
        at: datetime,
        note: str,
    ) -> ApprovalRecord:
        self._record_seq += 1
        record = ApprovalRecord(
            record_id=f"APR-{self._record_seq:06d}",
            change_id=request.change_id,
            version=request.version,
            approver=approver,
            decision=decision,
            at=at,
            note=note,
        )
        self._records[request.change_id].append(record)
        return record

    def _emit(
        self,
        request: ChangeRequest,
        kind: str,
        actor: str,
        at: datetime,
        detail: str = "",
    ) -> AuditEvent:
        self._event_seq += 1
        event = AuditEvent(
            event_id=f"AUD-{self._event_seq:06d}",
            change_id=request.change_id,
            version=request.version,
            kind=kind,
            actor=actor,
            at=at,
            detail=detail,
        )
        self._events[request.change_id].append(event)
        self._last_event_at[request.change_id] = at
        return event

    def _store(self, request: ChangeRequest) -> None:
        self._versions[request.change_id][request.version - 1] = request

    def _check_time(self, change_id: str, at: datetime) -> None:
        previous = self._last_event_at.get(change_id)
        if previous is not None and at < previous:
            raise TimeOrderViolation(
                f"event at {at.isoformat()} predates {previous.isoformat()}"
            )

    @staticmethod
    def _require_aware(at: datetime) -> None:
        if not isinstance(at, datetime) or at.tzinfo is None:
            raise TimeOrderViolation("timestamps must be timezone-aware datetimes")

    @staticmethod
    def _require_actor(actor: str) -> str:
        if not isinstance(actor, str) or not actor.strip():
            raise InvalidDeclaration("actor must be a non-empty string")
        return actor.strip()
