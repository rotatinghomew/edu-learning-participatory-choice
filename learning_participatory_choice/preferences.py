"""Pairwise analysis of incomplete course preferences.

No total ranking is constructed.  Each pair of courses that at least one
learner could compare is reported with its direct evidence (both courses were
rated/rejected by the same learner) and, separately, with any conclusion that
only appears when unanswered courses are filled in by an assumption.

Questionnaire frames are kept separate: a learner who was never shown a
course contributes no evidence about it, and learners from different
questionnaires are never pooled into a single population.  First-choice counts
are descriptive only; they are not treated as majority support.
"""

from collections import defaultdict
from itertools import combinations

POLICIES = ("missing", "unexpressed_lower")

POLICY_NOTES = {
    "missing": "未表达不等于排在最后；已展示但未表态的课程不进入比较证据。",
    "unexpressed_lower": "敏感性检查：假设已展示但未表态的课程低于所有明确排名的课程；"
                         "明确拒绝与未表态之间仍不比较。",
}


def _preferences_by_learner(dataset):
    result = defaultdict(dict)
    for pref in dataset.preferences:
        result[pref.learner_id][pref.course_id] = pref
    return result


def _shown_by_learner(dataset):
    return {learner_id: dataset.shown_courses(learner_id)
            for learner_id in dataset.responses_by_learner}


def _direct_outcome(pref_a, pref_b):
    """Compare two explicit expressions; returns a/b/tie, or None."""
    if pref_a is None or pref_b is None:
        return None
    if pref_a.choice == "rank" and pref_b.choice == "rank":
        if pref_a.rank < pref_b.rank:
            return "a"
        if pref_b.rank < pref_a.rank:
            return "b"
        return "tie"
    if pref_a.choice == "rank":
        return "a"
    if pref_b.choice == "rank":
        return "b"
    return "tie"  # both explicitly rejected = equally unacceptable


def _inferred_outcome(pref_a, pref_b, policy):
    """Compare when exactly one side is missing, under an explicit policy."""
    if policy != "unexpressed_lower":
        return None
    if pref_a is None and pref_b is None:
        return None
    if pref_a is None:
        return "b" if pref_b.choice == "rank" else None
    if pref_b is None:
        return "a" if pref_a.choice == "rank" else None
    return None


def _new_tally():
    return {"direct": {"a": 0, "b": 0, "tie": 0},
            "inferred": {"a": 0, "b": 0, "tie": 0},
            "missing": 0, "comparable": 0}


def _winner(counts):
    total = counts["a"] + counts["b"] + counts["tie"]
    if total == 0:
        return None
    if counts["a"] * 2 > total:
        return "a"
    if counts["b"] * 2 > total:
        return "b"
    return None


def _tally_pair(learner_ids, course_a, course_b, prefs, shown, policy):
    tally = _new_tally()
    for learner_id in learner_ids:
        visible = shown[learner_id]
        if course_a not in visible or course_b not in visible:
            continue
        tally["comparable"] += 1
        pref_a = prefs[learner_id].get(course_a)
        pref_b = prefs[learner_id].get(course_b)
        direct = _direct_outcome(pref_a, pref_b)
        if direct is not None:
            tally["direct"][direct] += 1
            continue
        inferred = _inferred_outcome(pref_a, pref_b, policy)
        if inferred is not None:
            tally["inferred"][inferred] += 1
        else:
            tally["missing"] += 1
    return tally


def _status_from_tally(tally, course_a, course_b):
    direct = tally["direct"]
    inferred = tally["inferred"]
    direct_total = sum(direct.values())
    inferred_total = sum(inferred.values())
    combined = {key: direct[key] + inferred[key] for key in ("a", "b", "tie")}
    direct_winner = _winner(direct)
    combined_winner = _winner(combined)
    winner = None
    if combined_winner is not None:
        winner = course_a if combined_winner == "a" else course_b

    if direct_total:
        basis = "direct_majority" if direct_winner else "direct_no_majority"
    elif inferred_total:
        basis = "inferred_majority" if combined_winner else "inferred_no_majority"
    else:
        basis = "no_evidence"

    total = sum(combined.values())
    return {
        "courses": [course_a, course_b],
        "winner": winner,
        "basis": basis,
        "direct": dict(direct),
        "inferred": dict(inferred),
        "missing_learner_count": tally["missing"],
        "unanswered_learner_count": tally["missing"],
        "comparable_learner_count": tally["comparable"],
        "direct_evidence_count": direct_total,
        "tie_share": (combined["tie"] / total) if total else None,
        "direct_winner": (course_a if direct_winner == "a" else course_b)
        if direct_winner else None,
        "decisive_only_with_assumption":
            bool(inferred_total) and combined_winner != direct_winner,
        "no_direct_evidence": direct_total == 0,
    }


def _first_choice_by_course(dataset, learner_ids, prefs, shown):
    """Count how many learners put each course in their tied-best set."""
    evidence = defaultdict(lambda: {
        "first_choice_intentions": 0, "strict_first_choice": 0,
        "tied_first_choice": 0, "ranked_by": 0, "rejected_by": 0,
        "unanswered_by": 0, "first_choice_learners": [],
    })
    for learner_id in learner_ids:
        visible = shown[learner_id]
        learner_prefs = prefs[learner_id]
        for course_id in visible:
            pref = learner_prefs.get(course_id)
            if pref is None:
                evidence[course_id]["unanswered_by"] += 1
            elif pref.choice == "reject":
                evidence[course_id]["rejected_by"] += 1
            else:
                evidence[course_id]["ranked_by"] += 1
        ranked = {cid: pref.rank for cid, pref in learner_prefs.items()
                  if cid in visible and pref.choice == "rank"}
        if not ranked:
            continue
        best = min(ranked.values())
        firsts = sorted(cid for cid, rank in ranked.items() if rank == best)
        for course_id in firsts:
            entry = evidence[course_id]
            entry["first_choice_intentions"] += 1
            entry["first_choice_learners"].append(learner_id)
            if len(firsts) == 1:
                entry["strict_first_choice"] += 1
            else:
                entry["tied_first_choice"] += 1
    return {course_id: evidence[course_id] for course_id in sorted(evidence)}


def _strong_components(nodes, edges):
    """Tarjan's algorithm; edges maps node -> set(nodes)."""
    index = {}
    low = {}
    stack = []
    on_stack = set()
    result = []
    counter = [0]

    def visit(node):
        index[node] = low[node] = counter[0]
        counter[0] += 1
        stack.append(node)
        on_stack.add(node)
        for neighbour in sorted(edges.get(node, ())):
            if neighbour not in index:
                visit(neighbour)
                low[node] = min(low[node], low[neighbour])
            elif neighbour in on_stack:
                low[node] = min(low[node], index[neighbour])
        if low[node] == index[node]:
            component = []
            while True:
                item = stack.pop()
                on_stack.remove(item)
                component.append(item)
                if item == node:
                    break
            result.append(sorted(component))

    for node in sorted(nodes):
        if node not in index:
            visit(node)
    return result


def _cycles(pair_statuses):
    edges = defaultdict(set)
    nodes = set()
    for (course_a, course_b), status in pair_statuses.items():
        nodes.update((course_a, course_b))
        if status["winner"] == course_a:
            edges[course_a].add(course_b)
        elif status["winner"] == course_b:
            edges[course_b].add(course_a)
    return [component for component in _strong_components(nodes, edges)
            if len(component) > 1]


def _analyse_block(dataset, block_id, title, learner_ids, prefs, shown, policy):
    visible = sorted({cid for learner_id in learner_ids for cid in shown[learner_id]})
    pairs = {}
    for course_a, course_b in combinations(visible, 2):
        tally = _tally_pair(learner_ids, course_a, course_b, prefs, shown, policy)
        pairs[(course_a, course_b)] = _status_from_tally(tally, course_a, course_b)
    return {
        "questionnaire_id": block_id,
        "title": title,
        "learner_count": len(learner_ids),
        "visible_courses": visible,
        "pairs": pairs,
        "cycles": _cycles(pairs),
        "first_choice": _first_choice_by_course(dataset, learner_ids, prefs, shown),
    }


def _pair_populations(block_results):
    """Per pair, show whether different questionnaire frames disagree."""
    pooled = defaultdict(list)
    for block in block_results:
        for key, status in block["pairs"].items():
            pooled[key].append({
                "questionnaire_id": block["questionnaire_id"],
                "winner": status["winner"],
                "basis": status["basis"],
                "direct": status["direct"],
                "inferred": status["inferred"],
                "missing_learner_count": status["missing_learner_count"],
                "comparable_learner_count": status["comparable_learner_count"],
            })
    result = {}
    for key, blocks in pooled.items():
        directions = {item["winner"] for item in blocks if item["winner"]}
        result[key] = {
            "courses": list(key),
            "population_dependent": len(directions) > 1,
            "blocks": blocks,
        }
    return result


def analyze_preferences(dataset, policy="missing"):
    """Return block-wise pairwise evidence, descriptives, and cycles."""
    if policy not in POLICIES:
        raise ValueError(f"未知的未表达偏好处理方式: {policy}")
    prefs = _preferences_by_learner(dataset)
    shown = _shown_by_learner(dataset)

    blocks = []
    for questionnaire in dataset.questionnaires:
        learner_ids = sorted(item.learner_id for item in dataset.responses
                             if item.questionnaire_id == questionnaire.questionnaire_id)
        blocks.append(_analyse_block(dataset, questionnaire.questionnaire_id,
                                     questionnaire.title, learner_ids, prefs,
                                     shown, policy))

    all_learners = sorted(dataset.responses_by_learner)
    pooled_visible = sorted({cid for learner_id in all_learners for cid in shown[learner_id]})
    pooled_pairs = {}
    for course_a, course_b in combinations(pooled_visible, 2):
        tally = _tally_pair(all_learners, course_a, course_b, prefs, shown, policy)
        if tally["comparable"]:
            pooled_pairs[(course_a, course_b)] = _status_from_tally(
                tally, course_a, course_b)

    return {
        "policy": policy,
        "policy_note": POLICY_NOTES[policy],
        "questionnaires": blocks,
        "pairs": pooled_pairs,
        "pair_populations": _pair_populations(blocks),
        "cycles": _cycles(pooled_pairs),
        "first_choice": _first_choice_by_course(dataset, all_learners, prefs, shown),
        "method": {
            "criterion": "按问卷分别做成对多数比较；不制造总排名，循环原样保留。",
            "framing": "只比较同一问卷中同时展示给学员的课程；不同问卷的可见范围"
                       "不合并为单一总体，跨问卷的方向不一致会被标出。",
            "unanswered": POLICY_NOTES[policy],
            "ties": "同等排名记为同序；明确拒绝只表示不接受，"
                    "不等于对未展示或未表态课程排序。",
            "first_choice": "首选票数是描述统计，不是多数支持或胜负依据；"
                            "并列首位分别计入。",
        },
    }


def sensitivity_to_unanswered(dataset):
    """Compare the default evidence with the explicit 'unexpressed lower' rule."""
    default = analyze_preferences(dataset, policy="missing")
    assumed = analyze_preferences(dataset, policy="unexpressed_lower")
    changes = []
    for key, status in default["pairs"].items():
        other = assumed["pairs"].get(key)
        if other is None or status["winner"] == other["winner"]:
            continue
        changes.append({
            "courses": list(key),
            "default_winner": status["winner"],
            "assumed_winner": other["winner"],
            "direct_winner": status["direct_winner"],
            "missing_learner_count": status["missing_learner_count"],
        })
    return {
        "default": default,
        "with_assumption": assumed,
        "changed_pairs": changes,
        "note": "只有下列成对结论会随未表达偏好的补全规则改变；"
                "其余比较不依赖该假设。",
    }
