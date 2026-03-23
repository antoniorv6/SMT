import sys
import os
import random
import numpy as np

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from SynthGenerator import VerovioGenerator

def test_generate_full_page():
    # Mocking necessary parts for VerovioGenerator if needed, 
    # but let's try to use it directly if possible.
    # We need some source data.
    sources = ["antoniorv6/grandstaff-ekern"] # This is what's used in data.py
    
    print("Initializing VerovioGenerator...")
    try:
        gen = VerovioGenerator(sources=sources, split="train")
    except Exception as e:
        print(f"Failed to initialize generator: {e}")
        # import traceback
        # traceback.print_exc()
        return

    print("Testing generate_full_page_score with 3 systems (should trigger the enumerate bug)...")
    try:
        # We need to make sure we have enough beats for 3 systems
        x, y = gen.generate_full_page_score(max_systems=3, strict_systems=False)
        print(f"Generated image shape: {x.size}")
        print(f"Generated ground truth length: {len(y)}")
        
        # Count how many systems are actually in the transcription
        # Systems are separated by '<b>'
        num_systems = 1 + sum(1 for token in y if token == '<b>')
        print(f"Number of systems in transcription: {num_systems}")
        
        if num_systems < 3:
            print("BUG CONFIRMED: Fewer systems than expected in transcription!")
        else:
            print("No bug detected in transcription length (maybe it was 3 already?).")
            
    except Exception as e:
        print(f"Failed to generate score: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_generate_full_page()
