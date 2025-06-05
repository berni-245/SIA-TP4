import json
from src.Hopfield import HopfieldNN
import src.patterns as pat

with open('configs/hopfield_config.json', 'r') as f:
    config = json.load(f)

pattern_chars = [let.upper() for let in config["net_patterns"]]
hopfield = HopfieldNN(pat.get_patterns(pattern_chars), config["max_epochs"])

query_pattern = pat.add_gaussian_noise_to_pattern(pat.get_patterns([config["query_letter"].upper()]), config["gauss_noice"])

hopfield.set_query_pattern(query_pattern)
pattern_history = hopfield.run_until_converged_with_history() 

pat.pattern_history_to_gif(pattern_history, config["output_file"], frame_duration_ms=100, expected_pattern_char=config["query_letter"])
