import os
import json
from sklearn.model_selection import train_test_split

# Path to UNHD data (adjust if needed)
data_dir = './unhd_data/UNHD-Complete-Data'

# Collect all PNG files
png_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]

# Extract unique "form_names" (WWW_LLL, ignoring V variation)
all_forms = sorted(set('_'.join(f.split('_')[:-1]) for f in png_files))  # e.g., ['001_001', '001_002', ...]

# Simple random split: 80% train, 10% val, 10% test
train_forms, temp_forms = train_test_split(all_forms, test_size=0.2, random_state=42)
val_forms, test_forms = train_test_split(temp_forms, test_size=0.5, random_state=42)

# Save splits as .uttlist files (mimic IAM's aachen splits)
os.makedirs('./unhd_split', exist_ok=True)
with open('./unhd_split/train_val.uttlist', 'w') as f:
    f.write('\n'.join(train_forms))
with open('./unhd_split/validation.uttlist', 'w') as f:
    f.write('\n'.join(val_forms))
with open('./unhd_split/test.uttlist', 'w') as f:
    f.write('\n'.join(test_forms))

# Create writer dict: Map writer strings ('001', '002', ...) to 0,1,2,...
writers = sorted(set(f.split('_')[0] for f in png_files))
wr_dict = {writer: idx for idx, writer in enumerate(writers)}

# Save as writers_dict.json (one for all subsets, since writers are shared)
with open('./writers_dict.json', 'w') as f:
    json.dump(wr_dict, f)

print(f"Created splits: {len(train_forms)} train, {len(val_forms)} val, {len(test_forms)} test")
print(f"Created writers_dict.json with {len(wr_dict)} writers")