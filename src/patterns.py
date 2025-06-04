from typing import List
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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
    Converts a 25x1 boolean numpy array into a 5x5 ASCII string,
    where True maps to '*' and False maps to ' ' (space).
    
    Returns the ASCII representation as a single string with newlines.
    """
    assert matrix_25x1.shape in [(25,), (25, 1)], "Input must be a 25x1 or 25-length array"
    assert matrix_25x1.dtype == np.bool_, "Input array must be of boolean type"

    arr = matrix_25x1.flatten()
    chars = ['*' if x else ' ' for x in arr]
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

def pattern_history_to_gif(
    pattern_history: np.ndarray,
    gif_path: str,
    pixel_size: int = 40,
    frame_duration_ms: int = 300,
    expected_pattern_char: str = ""
):
    """
    Converts a boolean pattern history array into a pixel grid GIF with grid lines
    and iteration text including an optional label (e.g., target character).

    Args:
        pattern_history: A (25, n) boolean NumPy array where each column is a 5x5 pattern frame.
        gif_path: Path to save the output GIF.
        pixel_size: Size of each grid square in the output image.
        duration: Duration of each frame in milliseconds.
        label: Optional character label to include in each frame (e.g., 'A', 'B', etc.).
    """
    assert pattern_history.shape[0] == 25, "Expected pattern height of 25 (for 5x5 grid)"
    num_frames = pattern_history.shape[1]
    frames = []

    grid_size = 5
    image_width = pixel_size * grid_size
    label_height = int(pixel_size * 1.2)
    image_height = image_width + label_height  # extra space for label

    # Load monospaced font or fallback
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", int(pixel_size * 0.5))
    except IOError:
        font = ImageFont.load_default()

    for i in range(num_frames):
        frame_data = pattern_history[:, i].reshape((5, 5))
        img = Image.new("RGB", (image_width, image_height), "white")
        draw = ImageDraw.Draw(img)

        # Draw pixels
        for y in range(grid_size):
            for x in range(grid_size):
                color = (0, 0, 0) if frame_data[y, x] else (255, 255, 255)
                x0 = x * pixel_size
                y0 = y * pixel_size
                draw.rectangle([x0, y0, x0 + pixel_size, y0 + pixel_size], fill=color)

        # Draw grid lines
        for j in range(grid_size + 1):
            pos = j * pixel_size
            draw.line([(pos, 0), (pos, image_width)], fill="gray")
            draw.line([(0, pos), (image_width, pos)], fill="gray")

        # Add label and step info
        step_text = f"Step {i + 1}"
        expected_pattern_text = f"Expected: {expected_pattern_char}" if expected_pattern_char else None
        full_text = f"{step_text}\n{expected_pattern_text}" if expected_pattern_text else step_text
        text_position = (5, image_width + 5)
        draw.text(text_position, full_text, font=font, fill="black")

        frames.append(img)

    # Save animated gif
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0
    )

def add_gaussian_noise_to_pattern(pattern: np.ndarray, noise_level: float) -> np.ndarray:
    """
    Applies bit-flip noise to a boolean vector (row, column, or flat) of 25 elements.
    
    Args:
        pattern (np.ndarray): A boolean vector of shape (25,), (25, 1), or (1, 25).
        noise_level (float): A float from 0 to 1 indicating the proportion of bits to flip.
        
    Returns:
        np.ndarray: A noisy copy of the original pattern with some bits flipped, same shape as input.
    """
    assert 0 <= noise_level <= 1, "noise_level must be between 0 and 1"
    assert pattern.dtype == bool, "pattern must be of boolean dtype"
    assert pattern.shape in [(25,), (25, 1), (1, 25)], "pattern must be shape (25,), (25,1), or (1,25)"

    num_flips = int(np.round(noise_level * 25))
    if num_flips == 0:
        return pattern.copy()

    flat = pattern.ravel()  # get 1D view of 25 elements
    flip_indices = np.random.choice(25, num_flips, replace=False)

    noisy_flat = flat.copy()
    noisy_flat[flip_indices] = ~noisy_flat[flip_indices]

    return noisy_flat.reshape(pattern.shape)
