"""Portfolio selection with resources, commitments, and coverage.

Selection is made on whole portfolios, never by picking the most popular
courses and deleting them afterwards.  A portfolio is feasible only if:

* every committed course is included,
* no teacher is scheduled in the same slot twice,
* opening costs plus each used teacher's shared block cost fit the budget,
* and every included course can reach its minimum enrolment using learners
  who were shown it, ranked it, and can attend its slot.

Coverage counts distinct learners who obtain at least one acceptable course in
a slot they can attend.  Multiple intentions of the same learner never count
twice, even when several of their ranked courses run.
"""

from itertools import combinations

from .feasibility import (analyze_commitments, schedule_cost, solve_assignment,
                          teacher_slot_conflicts)

MAX_PORTFOLIOS = 4096


def _optional_courses(dataset):
    """Courses worth enumerating: any non-committed course someone ranked."""
    ranked = {pref.course_id for pref in dataset.preferences if pref.choice == "rank"}
    return sorted(course_id for course_id in dataset.courses_by_id
                  if course_id not in dataset.committed_course_ids and course_id in ranked)


def _enumerate_portfolios(dataset):
    committed = tuple(sorted(dataset.committed_course_ids))
    optional = _optional_courses(dataset)
    if 2 ** len(optional) > MAX_PORTFOLIOS:
        raise ValueError(f"可选课程过多（{len(optional)} 门，{2 ** len(optional)} 个组合），"
                         "请拆分征询或先按课程范围分组")
    for size in range(len(optional) + 1):
        for extra in combinations(optional, size):
            yield tuple(sorted(committed + extra))


def evaluate_portfolio(dataset, selected):
    """Resource and assignment facts for one portfolio."""
    selected = tuple(sorted(selected))
    conflicts = teacher_slot_conflicts(dataset, selected)
    cost = schedule_cost(dataset, selected)
    over_budget = cost > dataset.budget + 1e-9
    if not selected:
        # The empty schedule is feasible when nothing is committed.
        assignment = {"feasible": True, "coverage": 0, "utility": 0.0,
                      "assignments": {}, "course_counts": {}}
    elif not conflicts and not over_budget:
        assignment = solve_assignment(dataset, selected)
    else:
        assignment = None
    feasible = (not conflicts and not over_budget
                and assignment is not None and assignment.get("feasible", False))
    return {
        "courses": selected,
        "conflicts": conflicts,
        "cost": cost,
        "over_budget": over_budget,
        "assignment": assignment,
        "feasible": feasible,
    }


def _coverage(evaluation):
    assignment = evaluation["assignment"]
    if not assignment or not assignment.get("feasible", False):
        return 0
    return assignment["coverage"]


def _utility(evaluation):
    assignment = evaluation["assignment"]
    if not assignment or not assignment.get("feasible", False):
        return 0.0
    return assignment["utility"]


def _summary_row(evaluation):
    return {
        "courses": list(evaluation["courses"]),
        "cost": evaluation["cost"],
        "coverage": _coverage(evaluation),
        "utility": _utility(evaluation),
        "feasible": evaluation["feasible"],
        "conflicts": {f"{teacher}/{slot}": courses
                      for (teacher, slot), courses in evaluation["conflicts"].items()},
        "counts": evaluation["assignment"].get("course_counts", {})
        if evaluation["assignment"] and evaluation["assignment"].get("feasible") else {},
    }


def _pareto_frontier(feasible):
    """Portfolios not beaten on cost, coverage, and preference satisfaction.

    Sort by cost, then keep the running 2-D maxima over (coverage, utility).
    """
    ordered = sorted(feasible, key=lambda item: (
        item["cost"], -_coverage(item), -_utility(item), item["courses"]))
    frontier = []
    maxima = []
    for item in ordered:
        covered, utility = _coverage(item), _utility(item)
        dominated = any(other_covered >= covered and other_utility >= utility - 1e-9
                        for other_covered, other_utility in maxima)
        if not dominated:
            frontier.append(item)
            maxima.append((covered, utility))
            maxima = [pair for pair in maxima if not (
                covered >= pair[0] and utility >= pair[1] - 1e-9
                and (covered > pair[0] or utility > pair[1] + 1e-9))]
    return frontier


def _pick_alternatives(feasible):
    """A small set of genuinely different trade-offs, not a total ranking."""
    non_empty = [item for item in feasible if item["courses"]] or feasible
    by_coverage = sorted(non_empty, key=lambda item: (
        -_coverage(item), item["cost"], -_utility(item), item["courses"]))
    by_cost = sorted(non_empty, key=lambda item: (
        item["cost"], -_coverage(item), -_utility(item), item["courses"]))
    by_utility = sorted(non_empty, key=lambda item: (
        -_utility(item), item["cost"], -_coverage(item), item["courses"]))
    return {
        "最大可接受覆盖": by_coverage[0],
        "最低成本": by_cost[0],
        "最高偏好满足度": by_utility[0],
    }


def _committed_only(evaluations, dataset):
    committed = tuple(sorted(dataset.committed_course_ids))
    return next((item for item in evaluations if item["courses"] == committed), None)


def _students_who_ranked(dataset, course_id):
    return {pref.learner_id for pref in dataset.preferences
            if pref.course_id == course_id and pref.choice == "rank"}


def _acceptable_learners(dataset, course_id):
    slot = dataset.courses_by_id[course_id].slot_id
    return {pref.learner_id for pref in dataset.preferences
            if pref.course_id == course_id and pref.choice == "rank"
            and slot in dataset.responses_by_learner[pref.learner_id].attendable_slots}


def explain_exclusion(dataset, reference, course_id, first_choice_evidence,
                      feasible=None):
    """Trace why a course is absent from a portfolio.

    ``category`` says whether the reason comes from resources, an existing
    commitment, or preference evidence; ``reason`` names the concrete check.
    """
    selected = reference["courses"]
    if course_id in selected:
        raise ValueError(f"课程 {course_id} 已在方案中")
    course = dataset.courses_by_id[course_id]
    with_course = tuple(sorted(selected + (course_id,)))
    trial = evaluate_portfolio(dataset, with_course)
    committed = dataset.committed_course_ids

    detail = {
        "course_id": course_id,
        "title": course.title,
        "committed": course.committed,
        "first_choice": first_choice_evidence,
        "ranked_learner_count": len(_students_who_ranked(dataset, course_id)),
        "acceptable_learner_count": len(_acceptable_learners(dataset, course_id)),
        "min_enrollment": course.min_enrollment,
        "capacity": course.capacity,
        "marginal_cost": round(trial["cost"] - reference["cost"], 6),
    }

    if course.committed:
        detail.update(reason="existing_commitment", category="existing_commitment",
                      message="该课程是既有开设承诺，但参考方案未能同时满足资源与承诺；"
                              "需人工调整预算或承诺，程序不会自动取消承诺。")
        return detail

    if trial["conflicts"]:
        conflicts = []
        with_committed = False
        for (teacher, slot), courses in trial["conflicts"].items():
            others = [cid for cid in courses if cid != course_id]
            hits_commitment = any(cid in committed for cid in others)
            with_committed = with_committed or hits_commitment
            conflicts.append({"teacher_id": teacher, "slot_id": slot,
                              "with_courses": others,
                              "with_committed_course": hits_commitment})
        detail.update(reason="existing_commitment_conflict" if with_committed
                      else "teacher_slot_conflict",
                      category="existing_commitment" if with_committed else "resource",
                      conflicts=conflicts,
                      message=("加入该课程会与既有开设承诺争用同一教师的同一时段；"
                               "承诺优先保留。" if with_committed else
                               "加入该课程会让同一教师在相同时段承担两门课。"))
        return detail

    if trial["over_budget"]:
        detail.update(reason="budget", category="resource",
                      cost_with_course=trial["cost"], budget=dataset.budget,
                      over_by=round(trial["cost"] - dataset.budget, 6),
                      committed_cost=schedule_cost(dataset, committed),
                      message="加入该课程后的总成本（含教师共享块成本）超出预算。")
        return detail

    assignment = trial["assignment"]
    if assignment is None or not assignment.get("feasible", False):
        failure = (assignment or {}).get("reason", "unknown")
        base = reference["assignment"]
        base_covered = set(base["assignments"]) if base and base.get("feasible") else set()
        acceptable = _acceptable_learners(dataset, course_id)
        detail.update(already_covered_acceptors=len(acceptable & base_covered),
                      acceptable_learners=sorted(acceptable))
        if detail["acceptable_learner_count"] < course.min_enrollment:
            detail.update(reason="insufficient_acceptable_learners",
                          category="preference_evidence", detail=failure,
                          message="展示过、实际排过且能到该时段的学员人数低于最低开班人数；"
                                  "首选票数高不等于有足够学员能来上课。")
        else:
            detail.update(reason="enrollment_or_capacity", category="resource",
                          detail=failure,
                          message="加入该课程后，无法同时让各门入选课程达到最低人数；"
                                  f"排过该课的 {detail['already_covered_acceptors']} 名学员"
                                  "已在参考方案中获得其他课程，"
                                  "而一份意向不能同时计入两门课。")
        return detail

    base = reference["assignment"]
    base_covered = set(base["assignments"]) if base and base.get("feasible") else set()
    trial_covered = set(assignment["assignments"])
    marginal = len(trial_covered - base_covered)
    detail.update(reason="no_marginal_coverage" if marginal == 0 else "coverage_trade_off",
                  category="preference_evidence",
                  marginal_covered=marginal,
                  base_coverage=len(base_covered),
                  trial_coverage=len(trial_covered),
                  base_utility=_utility(reference),
                  trial_utility=_utility(trial))
    if marginal == 0:
        detail["message"] = ("加入该课程虽然可行，但没有让任何尚未被覆盖的学员获得可接受选择，"
                             "反而增加成本；较高的首选票数可能来自同一批学员的重复意向或并列首选。")
    else:
        detail["message"] = ("把该课程加入参考方案可以提高覆盖，但会挤占其他课程的最低人数；"
                             "参考方案是同等或更好覆盖下的取舍，不是对偏好的否定。")
    return detail


def _describe_infeasibility(dataset, committed_only, commitments):
    assignment = commitments["assignment"] or {}
    conflicts = []
    for (teacher, slot), courses in committed_only["conflicts"].items():
        conflicts.append({"teacher_id": teacher, "slot_id": slot, "courses": courses})
    return {
        "committed_courses": commitments["committed_courses"],
        "teacher_slot_conflicts": conflicts,
        "cost": committed_only["cost"],
        "budget": dataset.budget,
        "over_budget": committed_only["over_budget"],
        "budget_shortfall": max(0.0, committed_only["cost"] - dataset.budget),
        "enrollment_problem": assignment.get("reason"),
        "message": "既有承诺本身无法同时满足：请核对教师时段冲突、预算或可到课人数。",
    }


def _alternative_reason(name):
    return {
        "最大可接受覆盖": "在预算与承诺内最大化获得至少一门可接受课程的学员人数。",
        "最低成本": "在满足全部承诺与最低人数前提下成本最低的组合。",
        "最高偏好满足度": "让已覆盖学员的偏好序位总和最优（序位越靠前权重越高）。",
    }[name]


def build_plans(dataset, first_choice_evidence=None):
    """Enumerate feasible portfolios, compare them, and trace exclusions."""
    first_choice_evidence = first_choice_evidence or {}
    commitments = analyze_commitments(dataset)
    evaluations = [evaluate_portfolio(dataset, courses)
                   for courses in _enumerate_portfolios(dataset)]
    feasible = [item for item in evaluations if item["feasible"]]
    committed_only = _committed_only(evaluations, dataset)

    result = {
        "budget": dataset.budget,
        "commitments": commitments,
        "portfolio_count": len(evaluations),
        "feasible_count": len(feasible),
        "frontier": [_summary_row(item) for item in _pareto_frontier(feasible)],
        "alternatives": {},
        "exclusions": [],
        "reference": None,
        "infeasibility": None,
    }

    if committed_only is not None and not committed_only["feasible"]:
        result["infeasibility"] = _describe_infeasibility(dataset, committed_only, commitments)

    if not feasible:
        result["message"] = ("没有任何组合能同时满足既有承诺与资源限制；"
                             "冲突来源见 infeasibility，未自动取消任何承诺。")
        return result

    picks = _pick_alternatives(feasible)
    reference = picks["最大可接受覆盖"]
    seen = set()
    for name, evaluation in picks.items():
        signature = (tuple(evaluation["courses"]), _coverage(evaluation),
                     round(_utility(evaluation), 6), evaluation["cost"])
        if signature in seen:
            continue
        seen.add(signature)
        row = _summary_row(evaluation)
        row["why"] = _alternative_reason(name)
        result["alternatives"][name] = row
    if committed_only is not None:
        signature = (tuple(committed_only["courses"]), _coverage(committed_only),
                     round(_utility(committed_only), 6), committed_only["cost"])
        if signature not in seen:
            row = _summary_row(committed_only)
            row["why"] = "仅保留既有承诺，作为对照；不额外开设课程。"
            result["alternatives"]["仅既有承诺"] = row

    result["reference"] = _summary_row(reference)
    selected = set(reference["courses"])
    for course_id in sorted(dataset.courses_by_id):
        if course_id in selected:
            continue
        result["exclusions"].append(
            explain_exclusion(dataset, reference, course_id,
                              first_choice_evidence.get(course_id, {})))
    return result
