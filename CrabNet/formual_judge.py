from collections import Counter
import re
import math
from functools import reduce

def equivalent_formula(formula1, formula2):
    def get_counts(formula):
        elements_counts = re.findall('([A-Z][a-z]*)([0-9]*)', formula)
        counts = Counter()
        for element, count in elements_counts:
            counts[element] += int(count) if count else 1
        return counts

    def normalize_counts(counts):
        factor = reduce(math.gcd, counts.values())
        for element in counts:
            counts[element] //= factor
        return counts

    return normalize_counts(get_counts(formula1)) == normalize_counts(get_counts(formula2))

formula1 = 'CFe2Mn'
formula2 = 'C4Fe8Mn4'

print(equivalent_formula(formula1, formula2))  # Outputs: True
