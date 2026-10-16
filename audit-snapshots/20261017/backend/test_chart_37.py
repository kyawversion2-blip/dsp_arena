import csv

TIME_SLIDER = 3000.0   # Stretch 1 sec to 3000 ms
FREQ_SLIDER = 2500.0   # Frequency multiplier 

groups = {'A': {'time': [], 'freq': []}, 'B': {'time': [], 'freq': []}, 'C': {'time': [], 'freq': []}, 'D': {'time': [], 'freq': []}}

with open('/Users/kyawnandar/Documents/37_Files.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        g = row['Group']
        if g not in groups: continue
        t = float(row['Duration']) * TIME_SLIDER
        f_hz = float(row['Impact_Freq_Hz']) * FREQ_SLIDER
        groups[g]['time'].append(t)
        groups[g]['freq'].append(f_hz)

print("=== 37_Files.csv Stretched using Chart Multipliers ===")
print(f"Time Multiplier: x{TIME_SLIDER}")
print(f"Freq Multiplier: x{FREQ_SLIDER}\n")

for g in ['A', 'B', 'C', 'D']:
    t_list = groups[g]['time']
    f_list = groups[g]['freq']
    if not t_list: continue
    min_t, max_t = min(t_list), max(t_list)
    min_f, max_f = min(f_list), max(f_list)
    print(f"Group {g}:")
    print(f"  Time Boundary: {min_t:7.1f}  to  {max_t:7.1f}")
    print(f"  Freq Boundary: {min_f:7.1f}  to  {max_f:7.1f}")
    print("-" * 40)

