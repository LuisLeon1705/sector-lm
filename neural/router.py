"""Import-only routing utilities shared by data preparation and tests."""
from collections import Counter


def route_from_import_tokens(import_tokens, import_rules, general_id=12):
    """Route from a pre-extracted list of import identifiers; source body is never accepted."""
    scores = Counter()
    for name in import_tokens:
        rule = import_rules.get(name)
        if rule:
            domain, weight = rule
            scores[int(domain)] += int(weight)
    if not scores:
        return general_id
    best_score = max(scores.values())
    return min(domain for domain, score in scores.items() if score == best_score)
