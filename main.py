import numpy as np
from src.Hopfield import HopfieldNN
import src.patterns as pat

# pattern_chars = ['H', 'M', 'N', 'W']
pattern_chars = ['G', 'R', 'T', 'V']
hopfield = HopfieldNN(pat.get_patterns(pattern_chars), 100)

# G
# query_pattern = """
#  ****
# *    
# *  **
# *   *
#  *** 
# """
# query_pattern = """
#  ****
# ** **
# ** **
# * * *
#  ****
# """

# R
query_pattern = """
**** 
*   *
**** 
*  * 
*   *
"""

# T
# query_pattern = """
# *****
#   *  
#   *  
#   *  
#   *  
# """
# query_pattern = """
# *****
# * * *
#  ** *
#  ** *
# * * *
# """
 
# V
# query_pattern = """
# *   *
# *   *
# *   *
#  * * 
#   *  
# """
# query_pattern = """
# *****
# * * *
# *   *
#  *** 
# * *
# """

query_pattern = pat.add_gaussian_noise_to_pattern(pat.ascii_art_to_pattern(query_pattern), 0.3)

hopfield.set_query_pattern(query_pattern)
pattern_history = hopfield.run_until_converged_with_history(4)

pat.pattern_history_to_gif(pattern_history, "./results/output.gif", frame_duration_ms=100, expected_pattern_char='R')
 
# pattern_evolution = []
# hopfield.set_query_pattern(query_pattern)
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
#   print(pattern_to_ascii(np.where(pattern == 1, True, False)))
#
# # print(np.hstack(pattern_evolution))
