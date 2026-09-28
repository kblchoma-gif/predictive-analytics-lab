"""
System Usability Scale.
Reference: Brooke (1996).
"""

SUS_QUESTIONS = [
    "I think that I would like to use this system frequently.",
    "I found the system unnecessarily complex.",
    "I thought the system was easy to use.",
    "I think that I would need the support of a technical person "
    "to be able to use this system.",
    "I found the various functions in this system were well integrated.",
    "I thought there was too much inconsistency in this system.",
    "I would imagine that most people would learn to use this system "
    "very quickly.",
    "I found the system very cumbersome to use.",
    "I felt very confident using the system.",
    "I needed to learn a lot of things before I could get going "
    "with this system.",
]


def compute_sus_score(responses) -> float:
    """responses: list of 10 ints in [1,5] in the order above."""
    score = 0
    for i, r in enumerate(responses):
        score += (r - 1) if i % 2 == 0 else (5 - r)
    return score * 2.5


def interpret_sus(score: float) -> str:
    if score >= 85:
        return "Excellent"
    if score >= 72:
        return "Good"
    if score >= 68:
        return "Above average"
    if score >= 51:
        return "OK"
    return "Poor"