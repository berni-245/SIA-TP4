import numpy as np
from Hopfield import HopfieldNN
from patterns import ascii_art_to_pattern, get_patterns

hopfield = HopfieldNN(get_patterns(['G', 'R', 'T', 'V']), 100)
# hopfield = HopfieldNN(get_patterns(['H', 'M', 'N', 'W']), 100)

query_pattern = """
*   *
*   *
*****
*   *
*   *
"""

pattern_evolution = []
hopfield.set_query_pattern(ascii_art_to_pattern(query_pattern))
pattern_evolution.append(hopfield.query_pattern)
for _ in range(hopfield.max_iters):
    print(hopfield.energy())
    hopfield.pattern_next()
    pattern_evolution.append(hopfield.query_pattern)
    if hopfield.pattern_converged():
        break
else:
    print("Max iterations reached without finding a pattern match")
    exit(1)

match_idx = hopfield.pattern_match()
if match_idx < 0:
    print("No matching column found.")
    exit(1)
else:
    print(f"Match found in column {match_idx}")

print(np.hstack(pattern_evolution))

# asdf = []
# for i in range(0, 4):
#     for j in range(0, 4):
#         if i != j:
#             asdf.append(hopfield.patterns[:, i:i+1] - hopfield.patterns[:, j:j+1])
#
# print(np.hstack(asdf))
