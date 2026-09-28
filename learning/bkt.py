"""
Bayesian Knowledge Tracing for adaptive assessment.

Reference: Corbett & Anderson (1995).
Parameters are hand-tuned priors (EM fitting is out of scope).
"""
from dataclasses import dataclass, field


@dataclass
class ConceptState:
    name: str
    p_init: float = 0.3
    p_trans: float = 0.15
    p_slip: float = 0.10
    p_guess: float = 0.20
    p_mastery: float = field(init=False)

    def __post_init__(self):
        self.p_mastery = self.p_init

    def update(self, correct: bool) -> float:
        p = self.p_mastery
        if correct:
            num = p * (1 - self.p_slip)
            den = num + (1 - p) * self.p_guess
        else:
            num = p * self.p_slip
            den = num + (1 - p) * (1 - self.p_guess)
        posterior = num / max(den, 1e-9)
        self.p_mastery = posterior + (1 - posterior) * self.p_trans
        return self.p_mastery


class AdaptiveAssessment:
    """
    Mastery-based question selection.

    Questions must have: id, concept, difficulty ('easy'|'medium'|'hard').
    """

    def __init__(self, questions: list):
        self.questions = {q["id"]: q for q in questions}
        self.asked = set()
        self.concepts = {}
        for q in questions:
            if q["concept"] not in self.concepts:
                self.concepts[q["concept"]] = ConceptState(q["concept"])

    def next_question(self):
        available = [q for qid, q in self.questions.items()
                     if qid not in self.asked]
        if not available:
            return None

        def priority(q):
            mastery = self.concepts[q["concept"]].p_mastery
            if mastery < 0.4:
                target = "easy"
            elif mastery < 0.7:
                target = "medium"
            else:
                target = "hard"
            return (mastery, 0 if q["difficulty"] == target else 1)

        available.sort(key=priority)
        return available[0]

    def submit(self, question_id: str, correct: bool):
        q = self.questions[question_id]
        self.asked.add(question_id)
        self.concepts[q["concept"]].update(correct)

    def mastery_report(self) -> dict:
        return {
            name: round(state.p_mastery, 3)
            for name, state in self.concepts.items()
        }

    def is_complete(self, n_questions: int = 8) -> bool:
        return len(self.asked) >= n_questions