import csv
from datetime import datetime
from pathlib import Path

from .model import (Course, Dataset, Preference, Questionnaire, Response,
                    Slot, Teacher)


def _parse_time(value):
    try:
        return datetime.strptime(value, "%H:%M").time()
    except ValueError as error:
        raise ValueError(f"无法识别的时间: {value}") from error


def _read_rows(path, required):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return []
    missing = set(required) - set(rows[0])
    if missing:
        raise ValueError(f"缺少字段: {', '.join(sorted(missing))}")
    return rows


def _split(value):
    return tuple(item.strip() for item in value.split(";") if item.strip())


def _parse_bool(value):
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "是"}:
        return True
    if normalized in {"false", "0", "no", "否"}:
        return False
    raise ValueError(f"无法识别的布尔值: {value}")


def read_records(path):
    """Legacy observation reader kept for the original aggregate summary."""
    rows = _read_rows(path, {"period", "group", "count", "value"})
    return rows


def load_dataset(directory):
    directory = Path(directory)
    slots = tuple(
        Slot(row["slot_id"], row["day"], _parse_time(row["start"]), _parse_time(row["end"]))
        for row in _read_rows(directory / "slots.csv",
                              {"slot_id", "day", "start", "end"}))
    teachers = tuple(
        Teacher(row["teacher_id"], row["name"], float(row["block_cost"]))
        for row in _read_rows(directory / "teachers.csv",
                              {"teacher_id", "name", "block_cost"}))
    courses = tuple(
        Course(row["course_id"], row["title"], row["teacher_id"], row["slot_id"],
               int(row["capacity"]), int(row["min_enrollment"]),
               float(row["opening_cost"]), _parse_bool(row["committed"]))
        for row in _read_rows(directory / "courses.csv",
                              {"course_id", "title", "teacher_id", "slot_id", "capacity",
                               "min_enrollment", "opening_cost", "committed"}))
    questionnaires = tuple(
        Questionnaire(row["questionnaire_id"], row["title"], _split(row["course_ids"]))
        for row in _read_rows(directory / "questionnaires.csv",
                              {"questionnaire_id", "title", "course_ids"}))
    responses = tuple(
        Response(row["learner_id"], row["questionnaire_id"],
                 frozenset(_split(row["attendable_slots"])))
        for row in _read_rows(directory / "responses.csv",
                              {"learner_id", "questionnaire_id", "attendable_slots"}))
    preferences = tuple(
        Preference(row["learner_id"], row["course_id"], row["choice"],
                   int(row["rank"]) if row["rank"].strip() else None)
        for row in _read_rows(directory / "preferences.csv",
                              {"learner_id", "course_id", "choice", "rank"}))
    program = _read_rows(directory / "program.csv", {"key", "value"})
    settings = {row["key"]: row["value"] for row in program}
    dataset = Dataset(slots, teachers, courses, questionnaires, responses, preferences,
                      budget=float(settings.get("budget", 0)))
    validate_dataset(dataset)
    return dataset


def validate_dataset(dataset):
    slot_ids = set(dataset.slots_by_id)
    teacher_ids = set(dataset.teachers_by_id)
    course_ids = set(dataset.courses_by_id)
    questionnaire_ids = set(dataset.questionnaires_by_id)

    for slot in dataset.slots:
        if slot.start >= slot.end:
            raise ValueError(f"时段 {slot.slot_id} 的开始时间不早于结束时间")
    for course in dataset.courses:
        if course.teacher_id not in teacher_ids:
            raise ValueError(f"课程 {course.course_id} 引用了未知教师")
        if course.slot_id not in slot_ids:
            raise ValueError(f"课程 {course.course_id} 引用了未知时段")
        if course.capacity < 0 or course.min_enrollment < 0:
            raise ValueError(f"课程 {course.course_id} 的班额不能为负")
        if course.min_enrollment > course.capacity:
            raise ValueError(f"课程 {course.course_id} 的最低人数超过容量")
        if course.opening_cost < 0:
            raise ValueError(f"课程 {course.course_id} 的开设成本不能为负")

    for questionnaire in dataset.questionnaires:
        for course_id in questionnaire.course_ids:
            if course_id not in course_ids:
                raise ValueError(f"问卷 {questionnaire.questionnaire_id} 引用了未知课程")
        if len(set(questionnaire.course_ids)) != len(questionnaire.course_ids):
            raise ValueError(f"问卷 {questionnaire.questionnaire_id} 重复展示同一课程")

    seen_learners = set()
    for response in dataset.responses:
        if response.learner_id in seen_learners:
            raise ValueError(f"学员 {response.learner_id} 有多份答卷")
        seen_learners.add(response.learner_id)
        if response.questionnaire_id not in questionnaire_ids:
            raise ValueError(f"学员 {response.learner_id} 引用了未知问卷")
        unknown_slots = response.attendable_slots - slot_ids
        if unknown_slots:
            raise ValueError(f"学员 {response.learner_id} 引用了未知时段")

    seen_preferences = set()
    for pref in dataset.preferences:
        if pref.learner_id not in dataset.responses_by_learner:
            raise ValueError(f"偏好记录引用了没有答卷的学员 {pref.learner_id}")
        if pref.course_id not in course_ids:
            raise ValueError(f"偏好记录引用了未知课程 {pref.course_id}")
        if pref.course_id not in dataset.shown_courses(pref.learner_id):
            raise ValueError(f"学员 {pref.learner_id} 评价了问卷未展示的课程 {pref.course_id}")
        if pref.choice not in {"rank", "reject"}:
            raise ValueError(f"未知的偏好类型: {pref.choice}")
        if pref.choice == "rank" and (pref.rank is None or pref.rank < 1):
            raise ValueError(f"学员 {pref.learner_id} 对 {pref.course_id} 的排名无效")
        if pref.choice == "reject" and pref.rank is not None:
            raise ValueError(f"学员 {pref.learner_id} 的拒绝记录不应带排名")
        key = (pref.learner_id, pref.course_id)
        if key in seen_preferences:
            raise ValueError(f"学员 {pref.learner_id} 重复评价 {pref.course_id}")
        seen_preferences.add(key)

    for learner_id in dataset.responses_by_learner:
        # Rank numbers may repeat: sharing a number expresses an equal tie.
        # They must start at 1.  Gaps are allowed because a tie at rank 1 may
        # be followed by rank 3 (competition ranking) or rank 2 (dense ranking).
        ranks_used = [pref.rank for pref in dataset.preferences
                      if pref.learner_id == learner_id and pref.choice == "rank"]
        if ranks_used and min(ranks_used) != 1:
            raise ValueError(f"学员 {learner_id} 的排名没有从 1 开始")
