#!/usr/bin/env python3
"""
Combine all training examples into a single dataset file for Dorothy's training.
"""

import json
import os
from pathlib import Path

def combine_training_data():
    """Combine all individual training examples into a single dataset."""
    base_path = Path("/home/ashiedu/Documents/heterodyne_detector/Dorothy/")
    output_file = base_path / "combined_train.jsonl"
    
    # Training examples to combine
    example_files = [
        base_path / "train.jsonl",
        base_path / "ex2.jsonl", 
        base_path / "ex3.jsonl"
    ]
    
    with open(output_file, 'w') as outfile:
        for file_path in example_files:
            if file_path.exists():
                with open(file_path, 'r') as infile:
                    for line in infile:
                        if line.strip():  # Skip empty lines
                            outfile.write(line)
    
    print(f"✅ Combined training data saved to: {output_file}")
    print(f"   Total examples: {count_examples(output_file)}")

def count_examples(file_path):
    """Count the number of examples in a JSONL file."""
    count = 0
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                count += 1
    return count

if __name__ == "__main__":
    combine_training_data()