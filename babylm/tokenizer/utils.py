import numpy as np
import matplotlib.pyplot as plt
import pickle
import nltk 
from pathlib import Path
import sys
import logging 
logger = logging.getLogger(__name__)
base_path = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(base_path)

def read_text_from_file(file_path):
    abs_path = file_path
    try:
        with open(abs_path, 'r') as file:
            text = file.read()
        logger.info(f"Loaded text from file {abs_path}")
        return text
    except IOError:
        logger.error(f"Error: Could not read from file {abs_path}")

def save_text(string, file_path):
    try:
        with open(file_path, "w") as file:
            file.write(string)
        logger.info(f"Processed corpus saved to {file_path}")
    except IOError:
        logger.error(f"Error: Could not write to file {file_path}")

def remove_speaker_labels(line):
    """
    Remove speaker labels from a line of text.
    """
    if ':' in line:
        return line.split(':', 1)[1].strip()
    return line

def remove_section_headers(text):
    """
    Remove text between and including the patterns '= = =' from the text.
    """
    cleaned_text = []
    skip = False
    for line in text.splitlines():
        if '= = =' in line:
            skip = not skip
            continue
        if not skip:
            cleaned_text.append(line)
    return '\n'.join(cleaned_text)

def get_vocab_size(text):
    tokens = nltk.word_tokenize(text)
    freq_dist = nltk.FreqDist(tokens)
    return len(freq_dist)