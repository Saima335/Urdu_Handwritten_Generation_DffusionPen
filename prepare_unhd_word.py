import os
import json

# Paths (adjust if needed)
data_dir = './unhd_data/UNHD-Complete-Data'
train_uttlist = './unhd_split/test.uttlist'
writers_dict_path = './writers_dict.json'
output_file = './unhd_data/unhd_test_fixed.txt'

# Load writers dict
with open(writers_dict_path, 'r') as f:
    wr_dict = json.load(f)

# Load train forms from uttlist
with open(train_uttlist, 'r') as f:
    train_forms = [line.strip() for line in f.readlines() if line.strip()]

# Collect lines for unhd_train_val_fixed.txt
lines = []
all_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
for filename in all_files:
    name = filename.replace('.png', '')  # e.g., 001_001_1
    form_name = '_'.join(name.split('_')[:-1])  # e.g., 001_001
    
    if form_name not in train_forms:
        continue
    
    img_path = os.path.join(data_dir, filename)
    gt_path = os.path.join(data_dir, name + '.gt.txt')
    
    if not os.path.exists(gt_path):
        continue
    
    with open(gt_path, 'r', encoding='utf-8') as f:
        transcr = f.read().strip()
    
    writer_name = name.split('_')[0]
    writer_id = wr_dict.get(writer_name, -1)
    if writer_id == -1:
        continue
    
    # Format: img_path,writer_id,transcr
    lines.append(f"{img_path},{writer_id},{transcr}")

# Save to file
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'w', encoding='utf-8') as f:
    for line in lines:
        f.write(line + '\n')

print(f"Created {output_file} with {len(lines)} entries")