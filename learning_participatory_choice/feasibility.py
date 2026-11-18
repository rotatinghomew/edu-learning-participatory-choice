"""Assignment and schedule feasibility.

Coverage is defined on people, not on intentions: a learner contributes at
most one covered person, and only through one course they ranked and can
physically attend.  Course minimum enrolment and capacity are enforced with a
lower-bound circulation, so a selected course must actually be fillable.

The flow solver stores an explicit flow on each edge, initialised to the
edge's lower bound.  After the feasible circulation is found the auxiliary
super-source/super-sink and return edges are removed rather than merely
frozen; their flow is either accounted for by the circulation value or is not
part of any learner/course edge.
"""

from collections import defaultdict, deque

INF = 10 ** 9


class _Network:
    def __init__(self, node_count):
        self.node_count = node_count
        self.edges = []          # [tail, head, lower, upper, cost, flow]
        self.adjacency = [[] for _ in range(node_count)]

    def add_edge(self, tail, head, lower, upper, cost=0):
        if lower > upper:
            raise ValueError("lower bound exceeds capacity")
        index = len(self.edges)
        self.edges.append([tail, head, lower, upper, cost, lower])
        self.adjacency[tail].append((index, 1))
        self.adjacency[head].append((index, -1))
        return index

    def flow(self, index):
        return self.edges[index][5]

    def remove_edge(self, index):
        tail, head, _, _, _, flow = self.edges[index]
        self.edges[index] = [tail, head, 0, 0, 0, 0]
        self.adjacency[tail] = [(item, direction) for item, direction in self.adjacency[tail]
                                if item != index]
        self.adjacency[head] = [(item, direction) for item, direction in self.adjacency[head]
                                if item != index]
        return flow

    def _residual(self, index, direction):
        _, _, lower, upper, cost, flow = self.edges[index]
        if direction == 1:
            return upper - flow, cost
        return flow - lower, -cost

    def _push(self, index, direction, amount):
        self.edges[index][5] += direction * amount

    def max_flow(self, source, sink):
        total = 0
        while True:
            previous = [None] * self.node_count
            seen = [False] * self.node_count
            seen[source] = True
            queue = deque([source])
            while queue and not seen[sink]:
                node = queue.popleft()
                for index, direction in self.adjacency[node]:
                    capacity, _ = self._residual(index, direction)
                    if capacity <= 0:
                        continue
                    tail, head = self.edges[index][0], self.edges[index][1]
                    neighbour = head if direction == 1 else tail
                    if seen[neighbour]:
                        continue
                    seen[neighbour] = True
                    previous[neighbour] = (index, direction)
                    queue.append(neighbour)
            if not seen[sink]:
                return total
            push = INF
            node = sink
            while node != source:
                index, direction = previous[node]
                capacity, _ = self._residual(index, direction)
                push = min(push, capacity)
                tail, head = self.edges[index][0], self.edges[index][1]
                node = tail if direction == 1 else head
            node = sink
            while node != source:
                index, direction = previous[node]
                self._push(index, direction, push)
                tail, head = self.edges[index][0], self.edges[index][1]
                node = tail if direction == 1 else head
            total += push

    def _find_negative_cycle(self):
        distance = [0] * self.node_count
        previous = [None] * self.node_count
        changed = None
        for _ in range(self.node_count):
            changed = None
            for index, (tail, head, lower, upper, cost, flow) in enumerate(self.edges):
                if lower == upper == flow:
                    continue
                capacity, step_cost = self._residual(index, 1)
                if capacity > 0 and distance[head] > distance[tail] + step_cost:
                    distance[head] = distance[tail] + step_cost
                    previous[head] = (index, 1)
                    changed = head
                capacity, step_cost = self._residual(index, -1)
                if capacity > 0 and distance[tail] > distance[head] + step_cost:
                    distance[tail] = distance[head] + step_cost
                    previous[tail] = (index, -1)
                    changed = tail
            if changed is None:
                return None
        node = changed
        for _ in range(self.node_count):
            if previous[node] is None:
                return None
            index, direction = previous[node]
            tail, head = self.edges[index][0], self.edges[index][1]
            node = tail if direction == 1 else head
        cycle = []
        cursor = node
        while True:
            index, direction = previous[cursor]
            cycle.append((index, direction))
            tail, head = self.edges[index][0], self.edges[index][1]
            cursor = tail if direction == 1 else head
            if cursor == node:
                break
        return cycle

    def cancel_negative_cycles(self, limit=10000):
        for _ in range(limit):
            cycle = self._find_negative_cycle()
            if cycle is None:
                return
            push = min(self._residual(index, direction)[0]
                       for index, direction in cycle)
            for index, direction in cycle:
                self._push(index, direction, push)
        raise RuntimeError("未能消除负成本环")


def lower_bound_circulation(network, edges_with_lower, source, sink,
                            super_source, super_sink):
    """Add a return edge and super nodes; return True if a circulation exists.

    Each lower-bounded edge is already initialised at its lower bound.  A node
    whose lower-bound inflow exceeds its outflow has positive ``balance`` and
    receives make-up flow from the super source; a node with excess outflow is
    drained to the super sink.  The return edge closes the circulation.
    """
    balance = [0] * network.node_count
    for index in edges_with_lower:
        tail, head, lower = network.edges[index][0], network.edges[index][1], network.edges[index][2]
        balance[tail] -= lower
        balance[head] += lower
    return_edge = network.add_edge(sink, source, 0, INF)
    auxiliary = []
    demand = 0
    for node, value in enumerate(balance):
        if value > 0:
            auxiliary.append(network.add_edge(super_source, node, 0, value))
            demand += value
        elif value < 0:
            auxiliary.append(network.add_edge(node, super_sink, 0, -value))
    saturated = network.max_flow(super_source, super_sink)
    for index in auxiliary:
        network.remove_edge(index)
    network.remove_edge(return_edge)
    return saturated == demand


def acceptable_courses(dataset, learner_id, selected):
    """Courses this learner ranked, that were shown, and that fit their slots."""
    response = dataset.responses_by_learner[learner_id]
    shown = dataset.shown_courses(learner_id)
    preferences = dataset.preferences_for(learner_id)
    result = {}
    for course_id in selected:
        course = dataset.courses_by_id[course_id]
        pref = preferences.get(course_id)
        if course_id not in shown or pref is None or pref.choice != "rank":
            continue
        if course.slot_id not in response.attendable_slots:
            continue
        result[course_id] = pref.rank
    return result


def solve_assignment(dataset, selected):
    """Assign each learner at most one acceptable course.

    Primary objective: number of distinct learners covered.  Secondary: the
    sum of 1/rank over assignments.  Both course capacity and minimum
    enrolment are hard constraints.
    """
    selected = tuple(sorted(selected))
    learners = sorted(dataset.responses_by_learner)
    courses = [dataset.courses_by_id[course_id] for course_id in selected]

    learner_nodes = {learner_id: 1 + offset for offset, learner_id in enumerate(learners)}
    course_nodes = {course.course_id: 1 + len(learners) + offset
                    for offset, course in enumerate(courses)}
    node_count = 1 + len(learners) + len(courses)
    source, sink = node_count, node_count + 1
    super_source, super_sink = node_count + 2, node_count + 3
    network = _Network(node_count + 4)

    acceptable = {learner_id: acceptable_courses(dataset, learner_id, selected)
                  for learner_id in learners}
    for learner_id in learners:
        network.add_edge(source, learner_nodes[learner_id], 0, 1)
        for course_id, rank in acceptable[learner_id].items():
            network.add_edge(learner_nodes[learner_id], course_nodes[course_id],
                             0, 1, -(1000 // rank))
    course_edges = []
    for course in courses:
        course_edges.append(network.add_edge(course_nodes[course.course_id], sink,
                                             course.min_enrollment, course.capacity))

    if not lower_bound_circulation(network, course_edges, source, sink,
                                    super_source, super_sink):
        return {"feasible": False, "reason": "enrollment_shortfall"}

    network.max_flow(source, sink)
    network.cancel_negative_cycles()

    assignments = {}
    course_counts = defaultdict(int)
    utilities = {}
    for learner_id in learners:
        for index, direction in network.adjacency[learner_nodes[learner_id]]:
            if direction != 1:
                continue
            head = network.edges[index][1]
            course_id = next((cid for cid, node in course_nodes.items()
                              if node == head), None)
            if course_id is None or network.flow(index) <= 0:
                continue
            assignments[learner_id] = course_id
            course_counts[course_id] += 1
            utilities[learner_id] = 1.0 / acceptable[learner_id][course_id]
            break

    if len(assignments.values()) != len(set(assignments)):
        return {"feasible": False, "reason": "duplicate_assignment"}
    for course in courses:
        count = course_counts.get(course.course_id, 0)
        if not course.min_enrollment <= count <= course.capacity:
            return {"feasible": False, "reason": "capacity_violation",
                    "course_id": course.course_id}

    return {
        "feasible": True,
        "coverage": len(assignments),
        "utility": sum(utilities.values()),
        "assignments": assignments,
        "course_counts": dict(course_counts),
    }


def schedule_cost(dataset, selected):
    """Opening costs plus each used teacher's once-per-schedule block cost."""
    selected = tuple(sorted(selected))
    if not selected:
        return 0.0
    opening = sum(dataset.courses_by_id[course_id].opening_cost for course_id in selected)
    teachers = {dataset.courses_by_id[course_id].teacher_id for course_id in selected}
    blocks = sum(dataset.teachers_by_id[teacher_id].block_cost for teacher_id in teachers)
    return opening + blocks


def teacher_slot_conflicts(dataset, selected):
    by_teacher_slot = defaultdict(list)
    for course_id in sorted(selected):
        course = dataset.courses_by_id[course_id]
        by_teacher_slot[(course.teacher_id, course.slot_id)].append(course_id)
    return {key: value for key, value in by_teacher_slot.items() if len(value) > 1}


def analyze_commitments(dataset):
    """Check whether commitments alone are simultaneously satisfiable."""
    committed = sorted(dataset.committed_course_ids)
    conflicts = teacher_slot_conflicts(dataset, committed)
    cost = schedule_cost(dataset, committed)
    budget_shortfall = max(0.0, cost - dataset.budget)
    assignment = solve_assignment(dataset, committed) if not conflicts else None
    return {
        "committed_courses": committed,
        "committed_cost": cost,
        "budget": dataset.budget,
        "teacher_slot_conflicts": conflicts,
        "over_budget": budget_shortfall > 0,
        "budget_shortfall": budget_shortfall,
        "assignment": assignment,
        "feasible": not conflicts and not budget_shortfall > 0
                    and assignment is not None and assignment.get("feasible", False),
    }
