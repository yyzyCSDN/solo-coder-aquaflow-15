import unittest
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

from aquaflow.security.change import (
    RISK_HIGH,
    RISK_LOW,
    STATUS_EFFECTIVE,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_REJECTED,
    STATUS_SUPERSEDED,
    STATUS_WITHDRAWN,
    ApprovalRecord,
    ChangeApprovalService,
    ChangeNotFound,
    DuplicateApproval,
    InvalidDeclaration,
    InvalidState,
    TimeOrderViolation,
    UnauthorizedApprover,
)

UTC = timezone.utc
T0 = datetime(2026, 10, 2, 9, 0, tzinfo=UTC)


def at(minute: int, second: int = 0) -> datetime:
    return T0 + timedelta(minutes=minute, seconds=second)


def make_service(key: str = "pump.override_psi", initial: object = 60):
    service = ChangeApprovalService()
    service.register_config(key, initial=initial)
    return service


def submit(service, key="pump.override_psi", new_value=90, risk=RISK_HIGH,
           required=2, applicant="alice", scope="zone-3 booster pumps"):
    return service.submit(
        config_key=key,
        new_value=new_value,
        scope=scope,
        applicant=applicant,
        risk=risk,
        required_approvals=required,
        at=at(0),
        reason="maintenance window",
    )


class TwoPersonCountersignTests(unittest.TestCase):
    def test_takes_effect_only_after_two_distinct_approvals(self):
        service = make_service()
        change = submit(service)

        first = service.approve(change.change_id, "bob", at(1))
        self.assertFalse(first.quorum_reached)
        self.assertFalse(first.effective)
        self.assertEqual(first.approvals, 1)
        self.assertEqual(service.get(change.change_id).status, STATUS_PENDING)
        self.assertEqual(service.effective_value("pump.override_psi"), 60)

        second = service.approve(change.change_id, "carol", at(2))
        self.assertTrue(second.quorum_reached)
        self.assertTrue(second.effective)
        self.assertEqual(second.status, STATUS_EFFECTIVE)

        final = service.get(change.change_id)
        self.assertEqual(final.status, STATUS_EFFECTIVE)
        self.assertEqual(final.effective_at, at(2))
        self.assertEqual(service.effective_value("pump.override_psi"), 90)

    def test_approval_records_are_independent_timestamped_and_immutable(self):
        service = make_service()
        change = submit(service)
        service.approve(change.change_id, "bob", at(1), note="ok by ops")
        service.approve(change.change_id, "carol", at(2))

        records = service.approvals(change.change_id)
        self.assertEqual(len(records), 2)
        self.assertIsInstance(records[0], ApprovalRecord)
        self.assertNotEqual(records[0].record_id, records[1].record_id)
        self.assertEqual(records[0].at, at(1))
        self.assertEqual(records[1].at, at(2))
        self.assertEqual([r.approver for r in records], ["bob", "carol"])
        for record in records:
            self.assertEqual(record.change_id, change.change_id)
            self.assertEqual(record.version, 1)
        with self.assertRaises(FrozenInstanceError):
            records[0].approver = "mallory"  # type: ignore[misc]

    def test_history_is_append_only_audit_trail(self):
        service = make_service()
        change = submit(service)
        service.approve(change.change_id, "bob", at(1))
        service.approve(change.change_id, "carol", at(2))
        events = service.history(change.change_id)
        kinds = [event.kind for event in events]
        self.assertEqual(kinds, ["created", "approved", "approved", "effective"])
        self.assertTrue(all(event.at is not None for event in events))


class DuplicateApprovalTests(unittest.TestCase):
    def test_same_approver_cannot_countersign_twice(self):
        service = make_service()
        change = submit(service, required=2)
        service.approve(change.change_id, "bob", at(1))
        with self.assertRaises(DuplicateApproval):
            service.approve(change.change_id, "bob", at(2))

        # One person's double approval must never satisfy a two-person quorum.
        self.assertEqual(service.approvals_remaining(change.change_id), 1)
        self.assertEqual(len(service.approvals(change.change_id)), 1)
        self.assertEqual(service.get(change.change_id).status, STATUS_PENDING)
        self.assertEqual(service.effective_value("pump.override_psi"), 60)

        # ... and the audit trail only records the genuine first approval.
        kinds = [e.kind for e in service.history(change.change_id)]
        self.assertEqual(kinds, ["created", "approved"])

    def test_duplicate_approval_on_revision_is_independent_not_reuse(self):
        service = make_service()
        change = submit(service, required=2)
        service.approve(change.change_id, "bob", at(1))
        service.reject(change.change_id, "carol", at(2), "bad window")

        revision = service.revise(change.change_id, at(3), reason="reschedule")
        self.assertEqual(revision.version, 2)
        # v2 starts with zero approvals even though bob approved v1.
        self.assertEqual(len(service.approvals(change.change_id, version=2)), 0)
        self.assertEqual(service.approvals_remaining(change.change_id), 2)
        # bob approving the new version is a fresh, independent record.
        service.approve(change.change_id, "bob", at(4))
        self.assertEqual(len(service.approvals(change.change_id, version=2)), 1)
        # v1 record is still intact and not counted towards v2.
        v1 = service.approvals(change.change_id, version=1)
        self.assertEqual(len(v1), 1)
        self.assertNotEqual(v1[0].record_id,
                            service.approvals(change.change_id, version=2)[0].record_id)


class PostQuorumValidationFailureTests(unittest.TestCase):
    def test_concurrency_conflict_after_quorum_blocks_enactment(self):
        service = make_service(initial=60)
        change = submit(service, new_value=90, required=2)
        service.approve(change.change_id, "bob", at(1))

        # A different change lands on the same key before the quorum completes,
        # so this version's declared old_value is now stale.
        other = submit(service, new_value=70, risk=RISK_LOW, required=1,
                       applicant="dave", scope="emergency trim")
        service.approve(other.change_id, "erin", at(2))
        self.assertEqual(service.effective_value("pump.override_psi"), 70)

        result = service.approve(change.change_id, "carol", at(3))
        self.assertTrue(result.quorum_reached)
        self.assertFalse(result.effective)
        self.assertEqual(result.status, STATUS_FAILED)
        self.assertTrue(any("stale" in error for error in result.errors))

        self.assertEqual(service.get(change.change_id).status, STATUS_FAILED)
        self.assertEqual(service.effective_value("pump.override_psi"), 70)
        kinds = [e.kind for e in service.history(change.change_id)]
        self.assertEqual(kinds, ["created", "approved", "approved", "failed"])

    def test_validator_failure_after_quorum_blocks_enactment(self):
        service = ChangeApprovalService()
        service.register_config(
            "chlorine.setpoint_mgl",
            initial=1.0,
            validators=[
                lambda key, old, new, live:
                    "setpoint above safety ceiling" if new > 2.0 else None
            ],
        )
        change = service.submit(
            config_key="chlorine.setpoint_mgl",
            new_value=3.5,
            scope="all treatment trains",
            applicant="alice",
            risk=RISK_HIGH,
            required_approvals=2,
            at=at(0),
        )
        service.approve(change.change_id, "bob", at(1))
        result = service.approve(change.change_id, "carol", at(2))
        self.assertTrue(result.quorum_reached)
        self.assertFalse(result.effective)
        self.assertEqual(result.status, STATUS_FAILED)
        self.assertTrue(any("safety ceiling" in e for e in result.errors))
        self.assertEqual(service.effective_value("chlorine.setpoint_mgl"), 1.0)
        # Approvals are retained as evidence even though enactment failed.
        self.assertEqual(len(service.approvals(change.change_id)), 2)

    def test_applicant_self_approval_never_counts(self):
        service = make_service()
        change = submit(service, required=2)
        with self.assertRaises(UnauthorizedApprover):
            service.approve(change.change_id, "alice", at(1))
        service.approve(change.change_id, "bob", at(2))
        service.approve(change.change_id, "carol", at(3))
        self.assertEqual(service.get(change.change_id).status, STATUS_EFFECTIVE)


class RejectAndWithdrawAuditTests(unittest.TestCase):
    def test_rejection_is_recorded_and_terminal(self):
        service = make_service()
        change = submit(service, required=2)
        service.approve(change.change_id, "bob", at(1))
        rejected = service.reject(change.change_id, "carol", at(2),
                                  "outside maintenance window")
        self.assertEqual(rejected.status, STATUS_REJECTED)
        event = service.history(change.change_id)[-1]
        self.assertEqual(event.kind, "rejected")
        self.assertEqual(event.actor, "carol")
        self.assertEqual(event.detail, "outside maintenance window")
        with self.assertRaises(InvalidState):
            service.approve(change.change_id, "dave", at(3))
        with self.assertRaises(InvalidDeclaration):
            service.reject(change.change_id, "dave", at(3), reason="   ")

    def test_withdrawal_is_recorded_and_terminal(self):
        service = make_service()
        change = submit(service, required=2)
        withdrawn = service.withdraw(change.change_id, at(1),
                                     reason="postponed to next window")
        self.assertEqual(withdrawn.status, STATUS_WITHDRAWN)
        event = service.history(change.change_id)[-1]
        self.assertEqual(event.kind, "withdrawn")
        self.assertEqual(event.actor, "alice")
        self.assertEqual(event.detail, "postponed to next window")
        with self.assertRaises(InvalidState):
            service.approve(change.change_id, "bob", at(2))

        other = submit(service)
        with self.assertRaises(UnauthorizedApprover):
            service.withdraw(other.change_id, at(1), reason="x", actor="bob")


class RevisionIsolationTests(unittest.TestCase):
    def test_new_version_inherits_no_approvals_and_marks_old_superseded(self):
        service = make_service()
        change = submit(service, new_value=90, required=2)
        service.approve(change.change_id, "bob", at(1))
        service.approve(change.change_id, "carol", at(2))
        self.assertEqual(service.get(change.change_id).status, STATUS_EFFECTIVE)

        revision = service.revise(change.change_id, at(3), new_value=95,
                                  reason="tune after observation")
        self.assertEqual(revision.version, 2)
        self.assertEqual(revision.status, STATUS_PENDING)
        self.assertEqual(revision.old_value, 90)  # based on the now-live value
        self.assertEqual(service.approvals_remaining(change.change_id), 2)
        self.assertEqual(
            service.get(change.change_id, version=1).status, STATUS_SUPERSEDED
        )

        # One approval on v2 must not be enough; old approvals are not reused.
        result = service.approve(change.change_id, "bob", at(4))
        self.assertFalse(result.quorum_reached)
        self.assertEqual(service.effective_value("pump.override_psi"), 90)
        result = service.approve(change.change_id, "dave", at(5))
        self.assertTrue(result.effective)
        self.assertEqual(service.effective_value("pump.override_psi"), 95)

    def test_failed_version_can_be_revised_and_does_not_reuse_quorum(self):
        service = make_service(initial=60)
        change = submit(service, new_value=90, required=2)
        service.approve(change.change_id, "bob", at(1))
        # invalidate via concurrent effective change
        other = submit(service, new_value=70, risk=RISK_LOW, required=1,
                       applicant="dave", scope="trim")
        service.approve(other.change_id, "erin", at(2))
        failed = service.approve(change.change_id, "carol", at(3))
        self.assertEqual(failed.status, STATUS_FAILED)

        revision = service.revise(change.change_id, at(4), new_value=75)
        self.assertEqual(revision.version, 2)
        self.assertEqual(revision.old_value, 70)
        self.assertEqual(len(service.approvals(change.change_id, version=2)), 0)
        service.approve(change.change_id, "bob", at(5))
        result = service.approve(change.change_id, "carol", at(6))
        self.assertTrue(result.effective)
        self.assertEqual(service.effective_value("pump.override_psi"), 75)


class DeclarationAndTimeTests(unittest.TestCase):
    def test_declaration_must_include_required_fields(self):
        service = make_service()
        with self.assertRaises(InvalidDeclaration):
            service.submit(config_key="pump.override_psi", new_value=90,
                           scope="  ", applicant="alice", risk=RISK_HIGH,
                           required_approvals=2, at=at(0))
        with self.assertRaises(InvalidDeclaration):
            service.submit(config_key="pump.override_psi", new_value=90,
                           scope="z", applicant="alice", risk="extreme",
                           required_approvals=2, at=at(0))
        with self.assertRaises(InvalidDeclaration):
            service.submit(config_key="pump.override_psi", new_value=90,
                           scope="z", applicant="alice", risk=RISK_HIGH,
                           required_approvals=1, at=at(0))

    def test_unknown_change_and_naive_timestamp(self):
        service = make_service()
        with self.assertRaises(ChangeNotFound):
            service.get("CHG-9999")
        change = submit(service)
        with self.assertRaises(TimeOrderViolation):
            service.approve(
                change.change_id, "bob",
                datetime(2026, 10, 2, 8, 0),  # naive
            )
        with self.assertRaises(TimeOrderViolation):
            service.approve(change.change_id, "bob", at(-1))

    def test_noop_change_rejected(self):
        service = make_service(initial=60)
        with self.assertRaises(InvalidDeclaration):
            submit(service, new_value=60)


if __name__ == "__main__":
    unittest.main()
