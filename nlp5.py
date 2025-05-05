#!/usr/bin/env python3
"""
morphology_table.py

Builds and displays a table of morphological add/delete operations
using pandas.
"""

import pandas as pd

def build_morphology_table():
    """
    Constructs a pandas DataFrame illustrating common
    morphological add/delete operations.
    """
    data = [
        {
            "Operation": "Add Prefix",
            "Morpheme": "un-",
            "Base Word": "happy",
            "Result Word": "unhappy",
            "Explanation": "Adding the negative prefix 'un-' to 'happy'."
        },
        {
            "Operation": "Add Suffix",
            "Morpheme": "-er",
            "Base Word": "read",
            "Result Word": "reader",
            "Explanation": "Adding the agentive suffix '-er' to 'read'."
        },
        {
            "Operation": "Add Suffix",
            "Morpheme": "-ing",
            "Base Word": "play",
            "Result Word": "playing",
            "Explanation": "Adding the progressive suffix '-ing' to 'play'."
        },
        {
            "Operation": "Delete Suffix",
            "Morpheme": "-s",
            "Base Word": "cats",
            "Result Word": "cat",
            "Explanation": "Removing the plural suffix '-s' from 'cats'."
        },
        {
            "Operation": "Delete Suffix",
            "Morpheme": "-ed",
            "Base Word": "jumped",
            "Result Word": "jump",
            "Explanation": "Removing the past tense suffix '-ed' from 'jumped'."
        },
        {
            "Operation": "Add Suffix (Spelling Change)",
            "Morpheme": "-ed",
            "Base Word": "revise",
            "Result Word": "revised",
            "Explanation": "Dropping the 'e' before adding '-ed' to 'revise'."
        },
        {
            "Operation": "Add Suffix",
            "Morpheme": "-al",
            "Base Word": "nation",
            "Result Word": "national",
            "Explanation": "Adding the adjective‑forming suffix '-al' to 'nation'."
        },
    ]

    df = pd.DataFrame(data)
    return df

def main():
    df = build_morphology_table()
    # Print the full table without the pandas index
    print("\nMorphological Add/Delete Table\n")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
