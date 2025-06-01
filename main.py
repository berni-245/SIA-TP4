import numpy as np
from src.Hopfield import HopfieldNN
from src.patterns import ascii_art_to_pattern, get_patterns, pattern_history_to_gif

pattern_chars = ['G', 'R', 'T', 'V']
# pattern_chars = ['H', 'M', 'N', 'W']
hopfield = HopfieldNN(get_patterns(pattern_chars), 100)

# G
query_pattern = """
 ****
** **
** **
* * *
 ****
"""

# T
# query_pattern = """
# *****
# * * *
#  ** *
#  ** *
# * * *
# """
 
# V
# query_pattern = """
# *****
# * * *
# *   *
#  *** 
# * *
# """

hopfield.set_query_pattern(ascii_art_to_pattern(query_pattern))
pattern_history = hopfield.run_until_converged_with_history()

pattern_history_to_gif(pattern_history, "output.gif", frame_duration_ms=100, expected_pattern_char='G')
 
# pattern_evolution = []
# hopfield.set_query_pattern(ascii_art_to_pattern(query_pattern))
# pattern_evolution.append(hopfield.query_pattern)
# for _ in range(hopfield.max_iters):
#     print(hopfield.energy())
#     hopfield.pattern_next()
#     pattern_evolution.append(hopfield.query_pattern)
#     if hopfield.pattern_converged():
#         break
# else:
#     print("Max iterations reached without finding a pattern match")
#     exit(1)
#
# match_idx = hopfield.pattern_match()
# if match_idx < 0:
#     print("No matching pattern found.")
# else:
#     print(f"Match found in '{pattern_chars[match_idx]}'")
#
# for i, pattern in enumerate(pattern_evolution):
#   print(f'\n\nt: {i}\n')
#   print(pattern_to_ascii(pattern))
#
# # print(np.hstack(pattern_evolution))
