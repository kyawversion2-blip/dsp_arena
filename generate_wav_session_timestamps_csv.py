#!/usr/bin/env python3
"""
Generate a consolidated CSV of all WAV files evaluated by AI models,
paired with their original creation/recording timestamps and session identifiers.
"""

import os
import csv
import datetime
import tarfile
import zipfile
import xml.etree.ElementTree as ET

BACKUP_TAR = "/Users/kyawnandar/Downloads/documents_backup_20260922.tar.gz"
MAPPING_TXT = "/Users/kyawnandar/Documents/mapping.txt"
EXCEL_FILE = "/Users/kyawnandar/Documents/ultrasonic_experiment.xlsx"
LABELS_CSV = "/Users/kyawnandar/Documents/interactive-v2/dsp_arena/labels.csv"
OUTPUT_CSV = "/Users/kyawnandar/Documents/interactive-v2/dsp_arena/ai_model_evaluated_wav_sessions.csv"

def collect_records():
    rows = []

    # 1. Labels map
    labels_map = {}
    if os.path.exists(LABELS_CSV):
        with open(LABELS_CSV, 'r', encoding='utf-8') as f:
            r = csv.reader(f)
            next(r, None)
            for line in r:
                if len(line) >= 2:
                    labels_map[line[0].strip()] = line[1].strip()

    # 2. Extract timestamps from documents_backup_20260922.tar.gz
    if os.path.exists(BACKUP_TAR):
        try:
            with tarfile.open(BACKUP_TAR, 'r:gz') as tar:
                training_labels = {}
                try:
                    m_lbl = tar.getmember('Documents/TESTING/Training/labels.csv')
                    f_lbl = tar.extractfile(m_lbl)
                    r_lbl = csv.reader(f_lbl.read().decode('utf-8').splitlines())
                    next(r_lbl, None)
                    for l in r_lbl:
                        if len(l) >= 2:
                            training_labels[l[0].strip()] = l[1].strip()
                except Exception:
                    pass

                for m in tar.getmembers():
                    if m.name.endswith('.wav') and not os.path.basename(m.name).startswith('._'):
                        dt_str = datetime.datetime.fromtimestamp(m.mtime).strftime('%Y-%m-%d %H:%M:%S')
                        fname = os.path.basename(m.name)
                        
                        session_tag = "Unknown"
                        group_label = labels_map.get(fname, "?")
                        
                        if "TESTING/Testing" in m.name:
                            session_tag = "Session_A_Testing_Set (July 17)"
                        elif "TESTING/Training" in m.name:
                            session_tag = "Session_A_Training_Set (July 17)"
                            group_label = training_labels.get(fname, group_label)
                        elif "Documents/4.wav" in m.name:
                            session_tag = "Standalone_Reference_Sample"

                        rows.append({
                            "File_Name": fname,
                            "Session": session_tag,
                            "Original_Create_Time": dt_str,
                            "Timestamp_Epoch": round(m.mtime, 3),
                            "Ground_Truth_Group": group_label,
                            "File_Size_Bytes": m.size,
                            "Source_Origin": m.name
                        })
        except Exception as e:
            print(f"Error reading backup tar: {e}")

    # 3. Read Session B from mapping.txt (July 04, 2026)
    if os.path.exists(MAPPING_TXT):
        try:
            with open(MAPPING_TXT, 'r', encoding='utf-8') as f:
                for line in f:
                    if "->" in line:
                        parts = line.strip().split("->")
                        orig_fn = parts[0].strip()
                        rest = parts[1].strip()
                        
                        # Only include valid .wav files
                        if not orig_fn.endswith(".wav"):
                            continue

                        uuid_part = rest.split()[0] if rest.split() else ""
                        ts_start = rest.find("(")
                        ts_end = rest.find(")")
                        ts_str = rest[ts_start+1:ts_end] if ts_start != -1 and ts_end != -1 else ""
                        
                        group_val = "?"
                        for g in ["A", "B", "C", "D"]:
                            if f" {g} " in rest or rest.endswith(f" {g}"):
                                group_val = g
                                break
                        if group_val == "?" and orig_fn in labels_map:
                            group_val = labels_map[orig_fn]

                        rows.append({
                            "File_Name": orig_fn,
                            "Session": "Session_B_UUID_Blind_Test (July 04)",
                            "Original_Create_Time": ts_str,
                            "Timestamp_Epoch": "",
                            "Ground_Truth_Group": group_val,
                            "File_Size_Bytes": "",
                            "Source_Origin": f"mapping.txt -> {uuid_part}"
                        })
        except Exception as e:
            print(f"Error reading mapping.txt: {e}")

    # 4. Read Session C from ultrasonic_experiment.xlsx (July 24, 2026)
    if os.path.exists(EXCEL_FILE):
        try:
            with zipfile.ZipFile(EXCEL_FILE) as z:
                xml_data = z.read('xl/worksheets/sheet1.xml')
                tree = ET.fromstring(xml_data)
                ns = {'ns': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
                rows_xml = tree.findall('.//ns:row', ns)
                
                seen_excel_files = set()
                for r in rows_xml[1:]:
                    cells = []
                    for c in r.findall('ns:c', ns):
                        v = c.find('ns:v', ns)
                        ist = c.find('.//ns:t', ns)
                        if ist is not None and ist.text:
                            cells.append(ist.text.strip())
                        elif v is not None and v.text:
                            cells.append(v.text.strip())
                        else:
                            cells.append('')

                    if len(cells) >= 10:
                        fn = cells[0]
                        if not fn.endswith(".wav"):
                            continue
                        dt_time = f"{cells[8]} {cells[9]}"
                        if fn not in seen_excel_files:
                            seen_excel_files.add(fn)
                            rows.append({
                                "File_Name": fn,
                                "Session": "Session_C_Ultrasonic_Experiment (July 24)",
                                "Original_Create_Time": dt_time,
                                "Timestamp_Epoch": "",
                                "Ground_Truth_Group": labels_map.get(fn, "?"),
                                "File_Size_Bytes": "",
                                "Source_Origin": f"ultrasonic_experiment.xlsx ({cells[1]})"
                            })
        except Exception as e:
            print(f"Error reading ultrasonic_experiment.xlsx: {e}")

    return rows

def main():
    records = collect_records()
    print(f"Total collected records: {len(records)}")

    # Sort records by Session, then Original_Create_Time, then File_Name
    records.sort(key=lambda x: (x["Session"], str(x["Original_Create_Time"]), x["File_Name"]))

    fieldnames = [
        "File_Name",
        "Session",
        "Original_Create_Time",
        "Timestamp_Epoch",
        "Ground_Truth_Group",
        "File_Size_Bytes",
        "Source_Origin"
    ]

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"✅ Successfully wrote {len(records)} records to:")
    print(f"   {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
