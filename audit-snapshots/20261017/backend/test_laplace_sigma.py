import os
import glob
import numpy as np
from scipy.io import wavfile
import pandas as pd

DATASET_PATH = "/Users/kyawnandar/Downloads/2612"

def analyze_laplace():
    wav_files = glob.glob(os.path.join(DATASET_PATH, "*.wav"))
    results = []

    for wav_path in wav_files:
        basename = os.path.basename(wav_path)
        file_num = int(basename.split(".")[0])
        group = "Unknown"
        if file_num == 6: group = "A"
        elif file_num in [1, 25, 33, 34]: group = "C"
        elif file_num in [7, 17, 24, 32]: group = "D"
        else: group = "B"
        
        rate, audio_data = wavfile.read(wav_path)
        
        # Audio is already float32 [-1.0, 1.0]
        # Find peak
        peak_idx = np.argmax(np.abs(audio_data))
        
        # Calculate Laplace Sigma (Damping Factor)
        decay = np.abs(audio_data)[peak_idx:]
        if len(decay) > 1000:
            log_d = np.log(decay + 1e-6)
            # We fit a line to the log of the decay envelope
            # The slope is -sigma. So Laplace_Sigma = abs(slope) * RATE
            slope, _ = np.polyfit(np.arange(len(log_d)), log_d, 1)
            laplace_sigma = float(abs(slope) * rate)
        else:
            laplace_sigma = 0.0
            
        results.append({
            "File": basename,
            "Group": group,
            "Laplace_Sigma": laplace_sigma
        })

    df = pd.DataFrame(results)
    
    print("\n=== Laplace Damping Factor (Sigma) Analysis ===")
    
    avg = df.groupby("Group").agg({
        "Laplace_Sigma": ["mean", "min", "max", "std"],
    }).round(2)
    print("\n--- Average Laplace Sigma by Group ---")
    print(avg)
    
    print("\n--- Raw Data (Sorted by Laplace Sigma) ---")
    print(df.sort_values(by="Laplace_Sigma").to_string(index=False))

if __name__ == "__main__":
    analyze_laplace()
