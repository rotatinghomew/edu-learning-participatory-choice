"""Human-readable operations report for course inquiry and scheduling."""

import sys
from pathlib import Path

from .preferences import sensitivity_to_unanswered
from .records import load_dataset
from .scheduling import build_plans

DEFAULT_DATA = Path(__file__).parent.parent / "data"


def _fmt_courses(course_ids, dataset):
    return "、".join(f"{cid}({dataset.courses_by_id[cid].title})" for cid in course_ids)


def _print_preferences(dataset, policy, analysis, out):
    out.write("=" * 72 + "\n")
    out.write(f"一、偏好证据（未表达偏好的处理：{policy}）\n")
    out.write(analysis["policy_note"] + "\n\n")

    out.write("首选意向（描述统计，不作为多数支持或胜负依据）\n")
    for course_id, entry in analysis["first_choice"].items():
        course = dataset.courses_by_id[course_id]
        out.write(f"  {course_id} {course.title}: "
                  f"{entry['first_choice_intentions']} 人次首选"
                  f"（严格首位 {entry['strict_first_choice']}，并列首位 "
                  f"{entry['tied_first_choice']}）\n")

    out.write("\n按问卷范围比较（不同问卷的课程范围不合并为单一总体）\n")
    for block in analysis["questionnaires"]:
        out.write(f"  [{block['questionnaire_id']}] {block['title']}，"
                  f"{block['learner_count']} 份答卷，展示 "
                  f"{_fmt_courses(block['visible_courses'], dataset)}\n")
        if not block["pairs"]:
            out.write("    没有可比较的课程对。\n")
        for (course_a, course_b), status in block["pairs"].items():
            direct = status["direct"]
            inferred = status["inferred"]
            if status["winner"]:
                verdict = f"倾向 {status['winner']}"
            else:
                verdict = "无多数结论"
            basis = {
                "direct_majority": "直接证据多数",
                "direct_no_majority": "直接证据无多数",
                "inferred_majority": "仅在补全假设下成立",
                "inferred_no_majority": "补全后仍无多数",
                "no_evidence": "无直接证据",
            }[status["basis"]]
            out.write(f"    {course_a} vs {course_b}: {verdict}（{basis}）；"
                      f"直接 a={direct['a']} b={direct['b']} 平={direct['tie']}，"
                      f"补全 a={inferred['a']} b={inferred['b']} 平={inferred['tie']}，"
                      f"未表达 {status['missing_learner_count']} 人\n")
        for cycle in block["cycles"]:
            out.write(f"    循环保留：{' → '.join(cycle)} → {cycle[0]}，"
                      "不能用总排名消解。\n")

    if analysis["cycles"]:
        out.write("\n合并范围内出现的循环（仍保留）：\n")
        for cycle in analysis["cycles"]:
            out.write(f"  {' → '.join(cycle)} → {cycle[0]}\n")


def _print_plans(dataset, plans, out):
    out.write("\n" + "=" * 72 + "\n")
    out.write("二、可行组合与资源约束\n")
    out.write(f"预算 {plans['budget']:.0f}；枚举 {plans['portfolio_count']} 个组合，"
              f"其中 {plans['feasible_count']} 个可行。\n")
    commitments = plans["commitments"]
    out.write(f"既有承诺：{_fmt_courses(commitments['committed_courses'], dataset) or '无'}，"
              f"承诺成本 {commitments['committed_cost']:.0f}，"
              f"承诺本身{'可行' if commitments['feasible'] else '不可行'}。\n")

    if plans["infeasibility"]:
        info = plans["infeasibility"]
        out.write("  既有承诺冲突来源：\n")
        if info["teacher_slot_conflicts"]:
            for item in info["teacher_slot_conflicts"]:
                out.write(f"    教师 {item['teacher_id']} 时段 {item['slot_id']} 同时承担 "
                          f"{'、'.join(item['courses'])}\n")
        if info["over_budget"]:
            out.write(f"    承诺成本 {info['cost']:.0f} 超过预算 {info['budget']:.0f}\n")
        if info["enrollment_problem"]:
            out.write(f"    最低开班人数无法满足：{info['enrollment_problem']}\n")
        out.write("    程序不会静默取消承诺，需人工调整预算、时段或承诺。\n")

    if plans["feasible_count"] == 0:
        out.write("没有同时满足承诺与资源限制的组合。\n")
        return

    out.write("\n可比较的取舍方案（不是总排名）\n")
    for name, row in plans["alternatives"].items():
        out.write(f"  {name}：{_fmt_courses(row['courses'], dataset)}\n")
        out.write(f"    成本 {row['cost']:.0f}；按人计可接受覆盖 {row['coverage']} 人；"
                  f"偏好满足度 {row['utility']:.2f}\n")
        out.write(f"    说明：{row['why']}\n")

    out.write(f"\n排除理由追查（参考方案："
              f"{_fmt_courses(plans['reference']['courses'], dataset)}）\n")
    for item in plans["exclusions"]:
        evidence = item.get("first_choice") or {}
        intentions = evidence.get("first_choice_intentions", 0)
        out.write(f"  {item['course_id']} {item['title']}："
                  f"类别 {item['category']}，理由 {item['reason']}；"
                  f"首选意向 {intentions} 人次，"
                  f"展示过且能到课的排课学员 {item['acceptable_learner_count']} 人，"
                  f"最低人数 {item['min_enrollment']}。\n")
        out.write(f"    {item['message']}\n")


def _print_sensitivity(sensitivity, out):
    out.write("\n" + "=" * 72 + "\n")
    out.write("三、未表达偏好的敏感性\n")
    out.write(sensitivity["note"] + "\n")
    if not sensitivity["changed_pairs"]:
        out.write("  没有成对结论随补全规则改变；直接证据已足以决定这些比较。\n")
    for change in sensitivity["changed_pairs"]:
        out.write(f"  {change['courses'][0]} vs {change['courses'][1]}："
                  f"默认 {change['default_winner'] or '无结论'}，"
                  f"补全后 {change['assumed_winner'] or '无结论'}；"
                  f"直接证据倾向 {change['direct_winner'] or '无'}"
                  f"（未表达 {change['missing_learner_count']} 人）。\n")
    out.write("\n准则与局限：\n")
    for key, value in sensitivity["default"]["method"].items():
        out.write(f"  {key}: {value}\n")


def main(out=None):
    out = out or sys.stdout
    dataset = load_dataset(DEFAULT_DATA)
    inquiry = sensitivity_to_unanswered(dataset)
    analysis = inquiry["default"]
    plans = build_plans(dataset, first_choice_evidence=analysis["first_choice"])
    _print_preferences(dataset, "missing", analysis, out)
    _print_plans(dataset, plans, out)
    _print_sensitivity(inquiry, out)
    _print_limitations(out)


def _print_limitations(out):
    out.write("\n" + "=" * 72 + "\n")
    out.write("四、局限\n")
    out.write("  * 不要求数据产生唯一赢家；没有多数证据的比较不强行定向。\n")
    out.write("  * 覆盖按去重后的学员计数，因此“意向总数”不能当作受益人数。\n")
    out.write("  * 结论依赖问卷展示范围与可到课时段；"
              "未展示、未表态、无法到课都不等于反对。\n")
    out.write("  * 可行性只覆盖本原型建模的约束（预算、教师时段、班额、承诺）；"
              "场地、设备等未建模。\n")
    out.write("  * 枚举规模有上限；课程很多时应先按问卷范围分组再分别比较。\n")
    out.write("  * 所有记录均为合成数据，不用于对个人作资格判断。\n")


if __name__ == "__main__":
    main()
