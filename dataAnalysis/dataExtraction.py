import pandas as pd
from collections import defaultdict
import json
import os

# === Load your CSV file ===
df = pd.read_csv("labels.csv")  # Make sure labels.csv is in the same folder

# === Create output folder ===
output_folder = "json_outputs"
os.makedirs(output_folder, exist_ok=True)

# === STEP 1: Label Agreement Scenarios ===

def classify_label_agreement(row):
    r1_1, r1_2 = row['pid_1_rank1'], row['pid_1_rank2']
    r2_1, r2_2 = row['pid_2_rank1'], row['pid_2_rank2']

    #skip if labels are missing
    if pd.isna(r1_1) or pd.isna(r2_1):
        return None

    if r1_1 == r2_1 and r1_2 == r2_2:
        return "same_both_same_order"
    elif r1_1 == r2_1:
        return "same_first_only"
    elif r1_2 == r2_2:
        return "same_second_only"
    elif set([r1_1, r1_2]) == set([r2_1, r2_2]):
        return "same_both_diff_order"
    elif r1_1 in [r2_1, r2_2] or r1_2 in [r2_1, r2_2]:
        return "one_label_matches"
    else:
        return "no_match"

df['agreement_type'] = df.apply(classify_label_agreement, axis=1)

agreement_json = defaultdict(list)
for _, row in df.iterrows():
    category = row['agreement_type']
    if category:
        agreement_json[category].append(row['video_name'])

with open(os.path.join(output_folder, "label_agreement.json"), "w") as f:
    json.dump(agreement_json, f, indent=2)

print("✅ Saved: json_outputs/label_agreement.json")

# === STEP 2: None Label Usage ===

def classify_none_usage(row):
    r1_labels = [row['pid_1_rank1'], row['pid_1_rank2']]
    r2_labels = [row['pid_2_rank1'], row['pid_2_rank2']]

    r1_has_none = "None" in r1_labels
    r2_has_none = "None" in r2_labels

    if r1_has_none and r2_has_none:
        return "both_raters_used_none"
    elif r1_has_none or r2_has_none:
        return "one_rater_used_none"
    else:
        return None

df['none_label_usage'] = df.apply(classify_none_usage, axis=1)

none_json = defaultdict(list)
for _, row in df.iterrows():
    category = row['none_label_usage']
    if category:
        none_json[category].append(row['video_name'])

with open(os.path.join(output_folder, "none_label_usage.json"), "w") as f:
    json.dump(none_json, f, indent=2)

print("✅ Saved: json_outputs/none_label_usage.json")

# === STEP 3: Jittery Videos ===

def classify_jittery(row):
    jittery_1 = str(row['pid_1_jitteryFlag']).lower() == "true"
    jittery_2 = str(row['pid_2_jitteryFlag']).lower() == "true"

    if jittery_1 and jittery_2:
        return "both_raters_jittery"
    elif jittery_1 or jittery_2:
        return "one_rater_jittery"
    else:
        return None

df['jittery_status'] = df.apply(classify_jittery, axis=1)

jittery_json = defaultdict(list)
for _, row in df.iterrows():
    category = row['jittery_status']
    if category:
        jittery_json[category].append(row['video_name'])

with open(os.path.join(output_folder, "jittery_videos.json"), "w") as f:
    json.dump(jittery_json, f, indent=2)

print("✅ Saved: json_outputs/jittery_videos.json")
