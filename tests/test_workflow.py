import csv
import io
import tempfile
import unittest
from pathlib import Path

from learning_participatory_choice.analysis import summarize
from learning_participatory_choice.feasibility import (acceptable_courses,
                                                       analyze_commitments,
                                                       schedule_cost,
                                                       solve_assignment)
from learning_participatory_choice.operations import review
from learning_participatory_choice.preferences import (analyze_preferences,
                                                       sensitivity_to_unanswered)
from learning_participatory_choice.records import load_dataset, read_records
from learning_participatory_choice.scheduling import build_plans

DATA = Path(__file__).parent.parent / "data"


def _write(directory, name, header, rows):
    path = Path(directory) / name
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8")


class DatasetFixture:
    """Two questionnaires with different visible ranges and a preference cycle.

    Q1 shows A, B, D; Q2 shows A, B, C, D.  On Q2, P4 prefers A to B, P5
    prefers B to C, and P6 prefers C to A, so no total order exists.  D is a
    committed course and shares teacher T2 with C, so running both costs the
    T2 block once.
    """

    def __init__(self):
        self.rows = {
            "slots": (["slot_id", "day", "start", "end"],
                      [["S1", "周一", "09:00", "10:30"],
                       ["S2", "周一", "14:00", "15:30"],
                       ["S3", "周二", "09:00", "10:30"]]),
            "teachers": (["teacher_id", "name", "block_cost"],
                         [["T1", "甲老师", "100"], ["T2", "乙老师", "50"]]),
            "courses": (["course_id", "title", "teacher_id", "slot_id", "capacity",
                         "min_enrollment", "opening_cost", "committed"],
                        [["A", "课程A", "T1", "S1", "4", "2", "30", "false"],
                         ["B", "课程B", "T1", "S2", "4", "2", "30", "false"],
                         ["C", "课程C", "T2", "S2", "4", "2", "20", "false"],
                         ["D", "课程D", "T2", "S3", "4", "1", "20", "true"]]),
            "questionnaires": (["questionnaire_id", "title", "course_ids"],
                               [["Q1", "甲卷", "A;B;D"],
                                ["Q2", "乙卷", "A;B;C;D"]]),
            "responses": (["learner_id", "questionnaire_id", "attendable_slots"],
                          [["P1", "Q1", "S1;S2;S3"],
                           ["P2", "Q1", "S1;S2;S3"],
                           ["P3", "Q1", "S1;S2"],
                           ["P4", "Q2", "S1;S2;S3"],
                           ["P5", "Q2", "S2;S3"],
                           ["P6", "Q2", "S1;S2"]]),
            "preferences": (["learner_id", "course_id", "choice", "rank"],
                            [["P1", "A", "rank", "1"], ["P1", "D", "rank", "2"],
                             ["P2", "A", "rank", "1"], ["P2", "B", "rank", "2"],
                             ["P3", "A", "rank", "1"], ["P3", "B", "rank", "1"],
                             ["P4", "A", "rank", "1"], ["P4", "B", "rank", "2"],
                             ["P4", "D", "rank", "3"],
                             ["P5", "B", "rank", "1"], ["P5", "C", "rank", "2"],
                             ["P6", "C", "rank", "1"], ["P6", "A", "rank", "2"]]),
            "program": (["key", "value"], [["budget", "300"]]),
        }

    def __enter__(self):
        self.directory = tempfile.TemporaryDirectory()
        for name, (header, rows) in self.rows.items():
            _write(self.directory.name, f"{name}.csv", header, rows)
        return self.directory.name

    def __exit__(self, *exc):
        self.directory.cleanup()


class PreferenceEvidenceTests(unittest.TestCase):
    def test_cycles_are_reported_not_resolved(self):
        # On Q2: P4 ranks A above B, P5 ranks B above C, P6 ranks C above A.
        # Each pairwise majority is direct evidence, yet the three cannot be
        # arranged into one order.
        with DatasetFixture() as directory:
            dataset = load_dataset(directory)
            analysis = analyze_preferences(dataset)
            q2 = next(item for item in analysis["questionnaires"]
                      if item["questionnaire_id"] == "Q2")
            self.assertEqual(q2["pairs"][("A", "B")]["winner"], "A")
            self.assertEqual(q2["pairs"][("B", "C")]["winner"], "B")
            self.assertEqual(q2["pairs"][("A", "C")]["winner"], "C")
            self.assertIn(["A", "B", "C"], q2["cycles"])
            self.assertFalse(any(name in analysis for name in ("ranking", "order")))

    def test_unanswered_is_not_last_and_scope_is_respected(self):
        with DatasetFixture() as directory:
            dataset = load_dataset(directory)
            analysis = analyze_preferences(dataset)
            q1 = next(item for item in analysis["questionnaires"]
                      if item["questionnaire_id"] == "Q1")
            # C was never shown on Q1, so it cannot appear in Q1 comparisons.
            self.assertNotIn("C", q1["visible_courses"])
            self.assertNotIn(("A", "C"), q1["pairs"])
            # A is a strict first choice for P1, P2 and P4, and a tied first
            # choice for P3.  Nobody who never saw A is counted against it.
            evidence = analysis["first_choice"]["A"]
            self.assertEqual(evidence["first_choice_intentions"], 4)
            self.assertEqual(evidence["strict_first_choice"], 3)
            # P5 was shown A on Q2 but left it blank; that blank is not a vote.
            self.assertEqual(evidence["unanswered_by"], 1)

    def test_ties_are_counted_as_shared_first_place(self):
        with DatasetFixture() as directory:
            dataset = load_dataset(directory)
            analysis = analyze_preferences(dataset)
            self.assertEqual(analysis["first_choice"]["A"]["tied_first_choice"], 1)
            self.assertEqual(analysis["first_choice"]["B"]["tied_first_choice"], 1)

    def test_sensitivity_points_to_pairs_decided_only_by_assumption(self):
        with DatasetFixture() as directory:
            dataset = load_dataset(directory)
            sensitivity = sensitivity_to_unanswered(dataset)
            changed = {tuple(item["courses"]): item
                       for item in sensitivity["changed_pairs"]}
            # C vs D has no direct evidence at all; only the assumption that
            # blanks rank last produces a direction.
            self.assertIn(("C", "D"), changed)
            self.assertIsNone(changed[("C", "D")]["direct_winner"])
            self.assertEqual(changed[("C", "D")]["assumed_winner"], "C")
            # A vs B is decided by direct evidence; the assumption must not
            # flip or reverse it.
            self.assertNotIn(("A", "B"), changed)

    def test_unexpressed_course_is_not_a_rejection(self):
        with DatasetFixture() as directory:
            dataset = load_dataset(directory)
            analysis = analyze_preferences(dataset)
            # P4 was shown D and ranked it; P5/P6 were shown D and left it
            # blank.  Those blanks produce no comparison, and D is not pushed
            # to last place.
            q2 = next(item for item in analysis["questionnaires"]
                      if item["questionnaire_id"] == "Q2")
            pair = q2["pairs"][("B", "D")]
            self.assertEqual(pair["direct"], {"a": 1, "b": 0, "tie": 0})
            self.assertEqual(pair["unanswered_learner_count"], 2)
            self.assertEqual(analysis["pairs"][("B", "D")]["inferred"],
                             {"a": 0, "b": 0, "tie": 0})


class ResourceAndCommitmentTests(unittest.TestCase):
    def test_shared_teacher_cost_counts_once_per_schedule(self):
        with DatasetFixture() as directory:
            dataset = load_dataset(directory)
            # T1 (100) once + two opening costs (30 + 30); T2 (50) once + D's
            # opening cost (20) = 230.  Adding C reuses T2's block: +20 only.
            self.assertEqual(schedule_cost(dataset, ["A", "B"]), 160)
            self.assertEqual(schedule_cost(dataset, ["A", "B", "D"]), 230)
            self.assertEqual(schedule_cost(dataset, ["A", "B", "C", "D"]), 250)

    def test_same_teacher_same_slot_is_infeasible(self):
        with DatasetFixture() as directory:
            rows = list(csv.reader(io.StringIO((Path(directory) / "courses.csv")
                                               .read_text(encoding="utf-8"))))
            rows[2][3] = "S1"  # course B moves onto T1/S1, where A already is
            _write(directory, "courses.csv", rows[0], rows[1:])
            dataset = load_dataset(directory)
            plans = build_plans(dataset)
            conflict_rows = [item for item in plans["frontier"]
                             if "T1/S1" in item["conflicts"]]
            self.assertFalse(conflict_rows)
            excluded = {item["course_id"]: item for item in plans["exclusions"]}
            self.assertTrue(any(item["reason"] == "teacher_slot_conflict"
                                for item in excluded.values()))

    def test_commitment_conflict_is_reported_not_silently_dropped(self):
        with DatasetFixture() as directory:
            rows = list(csv.reader(io.StringIO((Path(directory) / "courses.csv")
                                               .read_text(encoding="utf-8"))))
            rows[2][7] = "true"  # B becomes committed, keeps T1/S2
            rows[3][7] = "true"  # C becomes committed at T2/S2
            rows[4][3] = "S2"    # D moves to T2/S2, clashing with committed C
            rows[4][7] = "true"
            _write(directory, "courses.csv", rows[0], rows[1:])
            dataset = load_dataset(directory)
            commitments = analyze_commitments(dataset)
            self.assertFalse(commitments["feasible"])
            self.assertIn(("T2", "S2"), commitments["teacher_slot_conflicts"])
            plans = build_plans(dataset)
            self.assertIsNotNone(plans["infeasibility"])
            self.assertEqual(plans["feasible_count"], 0)
            self.assertIn("D", plans["infeasibility"]["committed_courses"])
            self.assertIn("C", plans["infeasibility"]["committed_courses"])


class CoverageTests(unittest.TestCase):
    def test_one_learner_is_never_counted_twice(self):
        with DatasetFixture() as directory:
            dataset = load_dataset(directory)
            result = solve_assignment(dataset, ["A", "B", "C", "D"])
            # Feasibility is not required to see the no-double-count rule.
            if result.get("feasible"):
                self.assertEqual(result["coverage"],
                                 len(set(result["assignments"])))
                self.assertLessEqual(result["coverage"],
                                     len(dataset.responses_by_learner))
            else:
                self.assertEqual(result["reason"], "enrollment_shortfall")
            feasible = solve_assignment(dataset, ["A", "B", "C"])
            self.assertTrue(feasible["feasible"])
            self.assertEqual(feasible["coverage"],
                             len(set(feasible["assignments"])))
            self.assertEqual(feasible["coverage"], 6)

    def test_course_minimum_enrolment_is_enforced(self):
        with DatasetFixture() as directory:
            dataset = load_dataset(directory)
            # C and D compete for P5 and P6; A and B for others.  With all four
            # selected, C cannot reach 2 and the portfolio is rejected whole.
            self.assertFalse(solve_assignment(dataset, ["A", "B", "C", "D"])["feasible"])
            self.assertTrue(solve_assignment(dataset, ["C", "D"])["feasible"])

    def test_attendable_slots_filter_acceptability(self):
        with DatasetFixture() as directory:
            dataset = load_dataset(directory)
            self.assertNotIn("D", acceptable_courses(dataset, "P3", ["D"]))
            self.assertIn("A", acceptable_courses(dataset, "P3", ["A"]))


class PlanTests(unittest.TestCase):
    def test_sample_plans_explore_whole_portfolios(self):
        dataset = load_dataset(DATA)
        plans = build_plans(dataset)
        self.assertGreater(plans["feasible_count"], 1)
        self.assertEqual(plans["commitments"]["committed_courses"], ["C5", "C8"])
        # A frontier, not a single winner, and every alternative keeps commitments.
        self.assertGreater(len(plans["frontier"]), 1)
        for row in plans["frontier"]:
            self.assertTrue({"C5", "C8"}.issubset(set(row["courses"])))

    def test_high_first_choice_course_has_a_traceable_exclusion(self):
        dataset = load_dataset(DATA)
        analysis = analyze_preferences(dataset)
        plans = build_plans(dataset, analysis["first_choice"])
        exclusions = {item["course_id"]: item for item in plans["exclusions"]}
        for course_id, item in exclusions.items():
            self.assertIn(item["category"],
                          {"resource", "existing_commitment", "preference_evidence"})
            self.assertEqual(
                item["first_choice"].get("first_choice_intentions", 0),
                analysis["first_choice"].get(course_id, {})
                .get("first_choice_intentions", 0))
            self.assertTrue(item["message"])
        # C2 has the most first-choice intentions in the sample data yet is
        # not in the reference portfolio; its exclusion must say why.
        self.assertEqual(
            analysis["first_choice"]["C2"]["first_choice_intentions"], 4)
        self.assertIn("C2", exclusions)
        self.assertEqual(exclusions["C2"]["category"], "resource")
        self.assertEqual(exclusions["C2"]["reason"], "enrollment_or_capacity")
        self.assertGreater(exclusions["C2"]["already_covered_acceptors"], 0)

    def test_plans_are_compared_as_whole_portfolios(self):
        dataset = load_dataset(DATA)
        plans = build_plans(dataset)
        # No feasible portfolio may drop a commitment, even if dropping it
        # would allow extra popular courses.
        for row in plans["frontier"]:
            self.assertTrue({"C5", "C8"}.issubset(set(row["courses"])))
        # The minimum-cost alternative is a different trade-off, not a subset
        # picked by popularity.
        cheapest = min(plans["frontier"], key=lambda row: row["cost"])
        self.assertLess(cheapest["cost"], plans["reference"]["cost"])

    def test_infeasible_commitments_are_explained(self):
        with DatasetFixture() as directory:
            rows = list(csv.reader(io.StringIO((Path(directory) / "courses.csv")
                                               .read_text(encoding="utf-8"))))
            rows[1][2] = "T2"    # A moves to T2/S1
            rows[1][3] = "S3"    # ... where D already sits, and both committed
            rows[1][7] = "true"
            rows[4][7] = "true"
            _write(directory, "courses.csv", rows[0], rows[1:])
            dataset = load_dataset(directory)
            plans = build_plans(dataset)
            self.assertEqual(plans["feasible_count"], 0)
            self.assertIsNotNone(plans["infeasibility"])
            self.assertEqual(
                plans["infeasibility"]["teacher_slot_conflicts"][0]["slot_id"], "S3")
            self.assertIn("message", plans["infeasibility"])
            self.assertIn("未自动取消任何承诺", plans["message"])


class LegacySummaryTests(unittest.TestCase):
    def test_legacy_aggregate_summary_still_works(self):
        rows = read_records(DATA / "observations.csv")
        summary = summarize(rows)
        self.assertEqual(len(summary), 4)
        self.assertEqual(sum(row["count"] for row in summary), 29)
        reviewed = review(summary, minimum_count=6)
        self.assertEqual([row["review"] for row in reviewed].count("资料不足"), 1)

    def test_missing_business_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("period,group,count\n2026-S1,甲,2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "value"):
                read_records(path)


if __name__ == "__main__":
    unittest.main()
