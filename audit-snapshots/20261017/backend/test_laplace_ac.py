import librosa
import numpy as np
import scipy.stats as stats

files = [
    ("/Users/kyawnandar/Documents/interactive-v2/blind_test_target/file_17.wav", "A"),
    ("/Users/kyawnandar/Documents/interactive-v2/blind_test_target/file_04.wav", "C"),
    ("/Users/kyawnandar/Documents/interactive-v2/blind_test_target/file_06.wav", "C"),
    ("/Users/kyawnandar/Documents/interactive-v2/blind_test_target/file_08.wav", "C"),
    ("/Users/kyawnandar/Documents/interactive-v2/blind_test_target/file_11.wav", "C"),
]

print("=== Checking if Laplace splits A and C ===")
for path, label in files:
    y, sr = librosa.load(path, sr=None)
    # Laplace calculation
    y_trim, _ = librosa.effects.trim(y, top_db=20)
    abs_y = np.abs(y_trim)
    env = np.convolve(abs_y, np.ones(int(0.01*sr))/int(0.01*sr), mode='same')
    
    # Fit Laplace
    loc, scale = stats.laplace.fit(env)
    
    print(f"Group {label} ({path.split('/')[-1]}): Laplace Scale (Sigma) = {scale:.6f}")

