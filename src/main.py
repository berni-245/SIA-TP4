from Hopfield import HopfieldNN
from patterns import ascii_art_to_pattern, get_patterns

hopfield = HopfieldNN(get_patterns(['J', 'B', 'D', 'Z']), 100)

query_pattern = """
 *** 
*   *
*****
*   *
*   *
"""

hopfield.find_pattern(ascii_art_to_pattern(query_pattern))
