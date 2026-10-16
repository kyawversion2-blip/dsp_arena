import os
import glob
import numpy as np
from scipy.io import wavfile
import pandas as pd
import scipy.signal

DATASET_PATH = "/Users/kyawnandar/Downloads/2612"

def analyze_decay():
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
        
        # Audio is already float32 [-1.0, 1.0] usually
        envelope = np.abs(audio_data)
        
        # Smooth envelope
        window_len = int(rate * 0.005)
        if window_len < 1: window_len = 1
        kernel = np.ones(window_len) / window_len
        smooth_env = scipy.signal.convolve(envelope, kernel, mode='same')
        
        max_amp = np.max(smooth_env)
        
        # Find peaks
        peaks, props = scipy.signal.find_peaks(
            smooth_env, 
            height=0.05 * max_amp, # lower threshold to catch smaller subsequent bounces
            distance=int(0.015 * rate)
        )
        
        if len(peaks) >= 2:
            peak_amps = smooth_env[peaks]
            # Sort peaks by time just in case, find_peaks already returns them sorted by time
            p1_amp = peak_amps[0]
            p2_amp = peak_amps[1]
            p3_amp = peak_amps[2] if len(peaks) >= 3 else 0.0
            
            decay_ratio_1_2 = p2_amp / p1_amp if p1_amp > 0 else 0
            decay_ratio_1_3 = p3_amp / p1_amp if p1_amp > 0 else 0
        else:
            p1_amp = max_amp
            p2_amp = 0
            p3_amp = 0
            decay_ratio_1_2 = 0
            decay_ratio_1_3 = 0

        results.append({
            "File": basename,
            "Group": group,
            "P1_Amp": p1_amp,
            "P2_Amp": p2_amp,
            "P3_Amp": p3_amp,
            "Decay_1_2": decay_ratio_1_2,
            "Decay_1_3": decay_ratio_1_3
        })

    df = pd.DataFrame(results)
    
    print("\n=== Decay Ratio Analysis (2612 Dataset) ===")
    
    avg = df.groupby("Group").agg({
        "Decay_1_2": ["mean", "min", "max", "std"],
        "Decay_1_3": ["mean", "min", "max"],
    }).round(4)
    print("\n--- Average Decay Ratios by Group ---")
    print(avg)
    
    print("\n--- Raw Data for B vs C vs A ---")
    print(df[['File', 'Group', 'Decay_1_2', 'Decay_1_3']].sort_values(by=['Group', 'File']).to_string(index=False))

if __name__ == "__main__":
    analyze_decay()
