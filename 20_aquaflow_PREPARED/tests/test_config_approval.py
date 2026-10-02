import unittest
from datetime import datetime, timedelta, timezone

from aquaflow.security.config_approval import (
    APPROVED,
    EFFECTIVE,
    PENDING,
    REJECTED,
    VALIDATION_FAILED,
    WITHDRAWN,
    ConfigChangeApprovalService,
)

T0 = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)


def at(minutes):
    return T0 + timedelta(minutes=minutes)


def submit(service, **overrides):
    params = dict(
        change_id="CHG-1",
        config_key="chlorine.setpoint_ppm",
        proposed_value=2.5,
        impact_scope=("treatment-train-1", "zone-north"),
        applicant="op.carol",
        risk_level="high",
        required_approvals=2,
        at=at(0),
    )
    params.update(overrides)
    return service.submit(**params)


class TwoPersonCountersignTest(unittest.TestCase):
    def test_quorum_then_activation_makes_change_effective(self):
        svc = ConfigChangeApprovalService()
        request = submit(svc)
        self.assertEqual(request.state, PENDING)
        self.assertEqual(request.version, 1)

        first = svc.approve("CHG-1", 1, "eng.alice", at(1))
        self.assertEqual(svc.get("CHG-1").state, PENDING)  # one approval is not a quorum
        with self.assertRaises(ValueError):
            svc.activate("CHG-1", 1, at(2))  # quorum not met -> cannot take effect
        self.assertIsNone(svc.current_config("chlorine.setpoint_ppm"))

        second = svc.approve("CHG-1", 1, "eng.bob", at(3))
        self.assertEqual(svc.get("CHG-1").state, APPROVED)

        # approval records are independent, timestamped and bound to version 1
        self.assertNotEqual(first.record_id, second.record_id)
        self.assertEqual([r.version for r in svc.approvals("CHG-1")], [1, 1])
        self.assertEqual([r.at for r in svc.approvals("CHG-1")], [at(1), at(3)])
        self.assertEqual({r.approver for r in svc.approvals("CHG-1")}, {"eng.alice", "eng.bob"})

        activated = svc.activate("CHG-1", 1, at(4))
        self.assertEqual(activated.state, EFFECTIVE)
        self.assertEqual(svc.current_config("chlorine.setpoint_ppm"), 2.5)

        actions = [event.action for event in svc.audit_trail("CHG-1")]
        self.assertEqual(actions, ["submit", "approve", "approve", "quorum_met", "effective"])
        seqs = [event.seq for event in svc.audit_trail()]
        self.assertEqual(seqs, sorted(seqs))


class DuplicateApprovalTest(unittest.TestCase):
    def test_same_approver_cannot_approve_twice(self):
        svc = ConfigChangeApprovalService()
        submit(svc)
        svc.approve("CHG-1", 1, "eng.alice", at(1))
        with self.assertRaises(ValueError):
            svc.approve("CHG-1", 1, "eng.alice", at(2))
        # the duplicate did not count: still one approval, quorum not met
        self.assertEqual(len(svc.approvals("CHG-1")), 1)
        self.assertEqual(svc.get("CHG-1").state, PENDING)
        with self.assertRaises(ValueError):
            svc.activate("CHG-1", 1, at(3))

    def test_applicant_cannot_self_approve(self):
        svc = ConfigChangeApprovalService()
        submit(svc)
        with self.assertRaises(ValueError):
            svc.approve("CHG-1", 1, "op.carol", at(1))


class ValidationFailureAfterQuorumTest(unittest.TestCase):
    def test_failed_validation_blocks_effect_and_forces_new_version(self):
        svc = ConfigChangeApprovalService()

        def within_envelope(request):
            if not 0.5 <= request.proposed_value <= 3.0:
                raise ValueError("setpoint outside safe envelope 0.5..3.0 ppm")

        svc.register_validator("chlorine.setpoint_ppm", within_envelope)
        submit(svc, proposed_value=4.0)
        svc.approve("CHG-1", 1, "eng.alice", at(1))
        svc.approve("CHG-1", 1, "eng.bob", at(2))
        self.assertEqual(svc.get("CHG-1").state, APPROVED)  # quorum met

        failed = svc.activate("CHG-1", 1, at(3))  # ...but validation fails
        self.assertEqual(failed.state, VALIDATION_FAILED)
        self.assertIsNone(svc.current_config("chlorine.setpoint_ppm"))  # never took effect
        events = svc.audit_trail("CHG-1")
        self.assertEqual(events[-1].action, "validation_failed")
        self.assertIn("safe envelope", events[-1].detail)
        with self.assertRaises(ValueError):
            svc.activate("CHG-1", 1, at(4))  # a failed version cannot be retried

        # fix the value in a new version; old approvals must not carry over
        revised = svc.revise("CHG-1", "op.carol", at(5), proposed_value=2.0)
        self.assertEqual((revised.version, revised.state), (2, PENDING))
        self.assertEqual(svc.approvals("CHG-1"), [])
        svc.approve("CHG-1", 2, "eng.alice", at(6))
        svc.approve("CHG-1", 2, "eng.bob", at(7))
        self.assertEqual(svc.activate("CHG-1", 2, at(8)).state, EFFECTIVE)
        self.assertEqual(svc.current_config("chlorine.setpoint_ppm"), 2.0)


class VersionIsolationTest(unittest.TestCase):
    def test_new_version_does_not_inherit_approvals(self):
        svc = ConfigChangeApprovalService()
        submit(svc)
        svc.approve("CHG-1", 1, "eng.alice", at(1))
        svc.approve("CHG-1", 1, "eng.bob", at(2))
        self.assertEqual(svc.get("CHG-1").state, APPROVED)

        revised = svc.revise("CHG-1", "op.carol", at(3), proposed_value=2.8)
        self.assertEqual((revised.version, revised.state), (2, PENDING))
        self.assertEqual(svc.approvals("CHG-1"), [])  # quorum must be rebuilt

        with self.assertRaises(ValueError):
            svc.approve("CHG-1", 1, "eng.dana", at(4))  # stale version cannot collect approvals
        with self.assertRaises(ValueError):
            svc.activate("CHG-1", 1, at(4))  # stale version cannot be activated
        with self.assertRaises(ValueError):
            svc.activate("CHG-1", 2, at(4))  # new version has no quorum yet

        # old records remain in history, still bound to version 1
        history = svc.decisions("CHG-1")
        self.assertEqual(len(history), 2)
        self.assertTrue(all(record.version == 1 for record in history))


class RejectWithdrawAuditTest(unittest.TestCase):
    def test_rejection_is_recorded_and_auditable(self):
        svc = ConfigChangeApprovalService()
        submit(svc)
        svc.approve("CHG-1", 1, "eng.alice", at(1))
        record = svc.reject("CHG-1", 1, "eng.bob", at(2), reason="setpoint risks low residual")
        self.assertEqual(record.decision, "reject")
        self.assertEqual((record.version, record.at), (1, at(2)))
        self.assertEqual(svc.get("CHG-1").state, REJECTED)

        events = svc.audit_trail("CHG-1")
        rejection = [e for e in events if e.action == "reject"]
        self.assertEqual(len(rejection), 1)
        self.assertEqual(rejection[0].actor, "eng.bob")
        self.assertEqual(rejection[0].at, at(2))
        self.assertIn("low residual", rejection[0].detail)

        with self.assertRaises(ValueError):
            svc.approve("CHG-1", 1, "eng.dana", at(3))  # rejected version is closed
        with self.assertRaises(ValueError):
            svc.reject("CHG-1", 1, "eng.dana", at(3), reason="too late")

    def test_withdrawal_is_recorded_and_auditable(self):
        svc = ConfigChangeApprovalService()
        submit(svc)
        svc.approve("CHG-1", 1, "eng.alice", at(1))
        with self.assertRaises(ValueError):
            svc.withdraw("CHG-1", "eng.alice", at(2), reason="not mine")  # only the applicant
        with self.assertRaises(ValueError):
            svc.withdraw("CHG-1", "op.carol", at(2), reason="")  # reason required

        withdrawn = svc.withdraw("CHG-1", "op.carol", at(2), reason="superseded by CHG-2")
        self.assertEqual(withdrawn.state, WITHDRAWN)
        events = svc.audit_trail("CHG-1")
        withdrawal = [e for e in events if e.action == "withdraw"]
        self.assertEqual(len(withdrawal), 1)
        self.assertEqual(withdrawal[0].actor, "op.carol")
        self.assertEqual(withdrawal[0].at, at(2))
        self.assertIn("superseded", withdrawal[0].detail)

        with self.assertRaises(ValueError):
            svc.approve("CHG-1", 1, "eng.bob", at(3))
        with self.assertRaises(ValueError):
            svc.activate("CHG-1", 1, at(3))


class DeclarationTest(unittest.TestCase):
    def test_submit_requires_full_declaration(self):
        svc = ConfigChangeApprovalService()
        with self.assertRaises(ValueError):
            submit(svc, change_id="C1", impact_scope=())  # impact scope must be declared
        with self.assertRaises(ValueError):
            submit(svc, change_id="C2", applicant="")  # applicant must be declared
        with self.assertRaises(ValueError):
            submit(svc, change_id="C3", risk_level="severe")  # unknown risk level
        with self.assertRaises(ValueError):
            submit(svc, change_id="C4", required_approvals=1)  # countersign needs >= 2
        with self.assertRaises(ValueError):
            submit(svc, change_id="C5", required_approvals=2.5)

    def test_time_reversal_rejected(self):
        svc = ConfigChangeApprovalService()
        submit(svc)
        with self.assertRaises(ValueError):
            svc.approve("CHG-1", 1, "eng.alice", at(-1))


if __name__ == "__main__":
    unittest.main()
