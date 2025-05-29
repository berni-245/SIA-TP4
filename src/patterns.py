from typing import List
import numpy as np

def _ascii_art_to_matrix(ascii_art: str) -> np.ndarray:
    """
    Convert a 5-line ASCII string into a 5x5 boolean matrix.
    '*' becomes True; all other characters become False.
    """
    lines = ascii_art.strip('\n').splitlines()
    if len(lines) != 5:
        raise ValueError("ASCII art must be exactly 5 lines.")

    matrix = np.zeros((5, 5), dtype=bool)
    for r, line in enumerate(lines):
        line = line.ljust(5)[:5]  # Ensure line is exactly 5 chars
        for c, ch in enumerate(line):
            matrix[r, c] = (ch == '*')
    return matrix

def ascii_art_to_pattern(ascii_art: str) -> np.ndarray:
    """
    Convert a 5-line ASCII string into a 15x1 boolean array.
    '*' becomes True; all other characters become False.
    """
    return _ascii_art_to_matrix(ascii_art).reshape(-1, 1)

def pattern_to_ascii(matrix_25x1: np.ndarray) -> str:
    """
    Converts a 25x1 numpy array of -1 and 1 into a 5x5 ASCII string,
    where -1 maps to ' ' (space) and 1 maps to '*'.
    
    Returns the ASCII representation as a single string with newlines.
    """
    assert matrix_25x1.shape == (25, 1) or matrix_25x1.shape == (25,), "Input must be a 25x1 or 25-length array"
    
    arr = matrix_25x1.flatten()
    chars = ['*' if x == 1 else ' ' for x in arr]
    rows = [''.join(chars[i*5:(i+1)*5]) for i in range(5)]
    return '\n'.join(rows)

def _parse_ascii_chars(filename):
    char_matrices = {}
    with open(filename, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if len(line) == 1 and line.isalpha():
            char = line
            i += 1
            ascii_block = ''.join(lines[i:i+5])
            i += 5
            char_matrices[char] = _ascii_art_to_matrix(ascii_block)
        else:
            i += 1

    return char_matrices

char_matrices = _parse_ascii_chars('data/patterns.txt')

def get_patterns(chars: List[str]) -> np.ndarray:
    # Create a list of reshaped column vectors
    columns = [char_matrices[char].reshape(-1, 1) for char in chars]
    
    # Concatenate column vectors horizontally
    patterns = np.hstack(columns)
    return patterns

