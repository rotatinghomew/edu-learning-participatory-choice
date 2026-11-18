"""In-memory records for course inquiry, feasibility, and scheduling.

The records deliberately keep four different facts apart:

* what a questionnaire showed a learner (the visible course range),
* what the learner expressed (ranked courses, ties, explicit rejection),
* what the learner could physically attend,
* what the programme can offer (teacher, slot, cost, capacity, commitments).

Keeping these separate is what prevents an unanswered course from being
silently treated as a last-place preference, and prevents a course that was
never shown from being treated as rejected.
"""

from dataclasses import dataclass, field
from datetime import time
from typing import Dict, FrozenSet, Optional, Tuple


@dataclass(frozen=True)
class Slot:
    slot_id: str
    day: str
    start: time
    end: time


@dataclass(frozen=True)
class Teacher:
    teacher_id: str
    name: str
    block_cost: float


@dataclass(frozen=True)
class Course:
    course_id: str
    title: str
    teacher_id: str
    slot_id: str
    capacity: int
    min_enrollment: int
    opening_cost: float
    committed: bool


@dataclass(frozen=True)
class Questionnaire:
    questionnaire_id: str
    title: str
    course_ids: Tuple[str, ...]


@dataclass(frozen=True)
class Response:
    learner_id: str
    questionnaire_id: str
    attendable_slots: FrozenSet[str]


@dataclass(frozen=True)
class Preference:
    learner_id: str
    course_id: str
    choice: str  # "rank" or "reject"
    rank: Optional[int]


@dataclass
class Dataset:
    slots: Tuple[Slot, ...]
    teachers: Tuple[Teacher, ...]
    courses: Tuple[Course, ...]
    questionnaires: Tuple[Questionnaire, ...]
    responses: Tuple[Response, ...]
    preferences: Tuple[Preference, ...]
    budget: float = 0.0

    slots_by_id: Dict[str, Slot] = field(default_factory=dict, init=False)
    teachers_by_id: Dict[str, Teacher] = field(default_factory=dict, init=False)
    courses_by_id: Dict[str, Course] = field(default_factory=dict, init=False)
    questionnaires_by_id: Dict[str, Questionnaire] = field(default_factory=dict, init=False)
    responses_by_learner: Dict[str, Response] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.slots_by_id = {item.slot_id: item for item in self.slots}
        self.teachers_by_id = {item.teacher_id: item for item in self.teachers}
        self.courses_by_id = {item.course_id: item for item in self.courses}
        self.questionnaires_by_id = {item.questionnaire_id: item for item in self.questionnaires}
        self.responses_by_learner = {item.learner_id: item for item in self.responses}

    @property
    def committed_course_ids(self) -> FrozenSet[str]:
        return frozenset(course.course_id for course in self.courses if course.committed)

    def shown_courses(self, learner_id: str) -> FrozenSet[str]:
        response = self.responses_by_learner[learner_id]
        questionnaire = self.questionnaires_by_id[response.questionnaire_id]
        return frozenset(questionnaire.course_ids)

    def preferences_for(self, learner_id: str) -> Dict[str, Preference]:
        return {pref.course_id: pref for pref in self.preferences if pref.learner_id == learner_id}
