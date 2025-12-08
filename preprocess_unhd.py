# import os
# import torch
# from PIL import Image
# import numpy as np
# from torchvision import transforms
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm

# # Paths
# data_dir = './unhd_data/UNHD-Complete-Data'  # Adjust to your unzip path
# output_dir = './saved_unhd_data'  # Mimic IAM structure
# os.makedirs(output_dir, exist_ok=True)

# # Transforms: Grayscale, resize height to 128, normalize
# transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),
#     transforms.Resize((128, None)),  # Resize height, keep aspect
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5], std=[0.5])  # Typical for diffusion
# ])

# # Collect data
# data = []
# for file in tqdm(os.listdir(data_dir)):
#     if file.endswith('.png'):
#         img_path = os.path.join(data_dir, file)
#         gt_path = os.path.join(data_dir, file.replace('.png', '.gt.txt'))
        
#         if not os.path.exists(gt_path):
#             continue  # Skip if no transcription
        
#         # Load image
#         img = Image.open(img_path)
#         img_tensor = transform(img)
        
#         # Load text (Urdu UTF-8)
#         with open(gt_path, 'r', encoding='utf-8') as f:
#             text = f.read().strip()
        
#         # Extract writer_id from filename (first 3 digits)
#         writer_id = file.split('_')[0]
        
#         data.append({
#             'img': img_tensor,
#             'text': text,
#             'writer_id': writer_id
#         })

# # Split: 80% train, 10% val, 10% test
# train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
# val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# # Save as .pt
# torch.save(train_data, os.path.join(output_dir, 'train.pt'))
# torch.save(val_data, os.path.join(output_dir, 'val.pt'))
# torch.save(test_data, os.path.join(output_dir, 'test.pt'))

# print(f"Processed {len(data)} samples. Saved to {output_dir}")

"""
UNHD Dataset Preprocessing Script
Creates train/val/test splits and data lists compatible with DiffusionPen
"""

import os
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def create_splits(data_dir='./unhd_data/UNHD-Complete-Data'):
    """Create train/val/test splits based on form IDs"""
    
    print("Step 1: Creating train/val/test splits...")
    
    # Collect all PNG files
    png_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
    print(f"Found {len(png_files)} PNG files")
    
    # Extract unique form names (WWW_LLL pattern)
    # Format: 001_001_1.png -> 001_001
    all_forms = sorted(set('_'.join(f.split('_')[:-1]) for f in png_files))
    print(f"Found {len(all_forms)} unique forms")
    
    # Split: 80% train, 10% val, 10% test
    train_forms, temp_forms = train_test_split(all_forms, test_size=0.2, random_state=42)
    val_forms, test_forms = train_test_split(temp_forms, test_size=0.5, random_state=42)
    
    print(f"Train forms: {len(train_forms)}")
    print(f"Val forms: {len(val_forms)}")
    print(f"Test forms: {len(test_forms)}")
    
    # Create output directory
    os.makedirs('./unhd_split', exist_ok=True)
    
    # Save splits as .uttlist files
    with open('./unhd_split/train_val.uttlist', 'w') as f:
        f.write('\n'.join(train_forms))
    
    with open('./unhd_split/validation.uttlist', 'w') as f:
        f.write('\n'.join(val_forms))
    
    with open('./unhd_split/test.uttlist', 'w') as f:
        f.write('\n'.join(test_forms))
    
    print("✓ Created split files in ./unhd_split/")
    
    return train_forms, val_forms, test_forms


def create_writer_dict(data_dir='./unhd_data/UNHD-Complete-Data'):
    """Create writer ID to index mapping"""
    
    print("\nStep 2: Creating writer dictionary...")
    
    # Collect all PNG files
    png_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
    
    # Extract unique writer IDs (first part before first underscore)
    # Format: 001_001_1.png -> 001
    writers = sorted(set(f.split('_')[0] for f in png_files))
    print(f"Found {len(writers)} unique writers")
    
    # Create mapping: writer_string -> index
    wr_dict = {writer: idx for idx, writer in enumerate(writers)}
    
    # Save as JSON
    with open('./writers_dict.json', 'w') as f:
        json.dump(wr_dict, f, indent=2)
    
    print(f"✓ Created writers_dict.json with {len(wr_dict)} writers")
    print(f"  Writer ID range: 0-{len(wr_dict)-1}")
    
    return wr_dict


def create_data_lists(data_dir='./unhd_data/UNHD-Complete-Data', 
                      train_forms=None, val_forms=None, test_forms=None,
                      wr_dict=None):
    """Create data list files (img_path,writer_id,transcription)"""
    
    print("\nStep 3: Creating data list files...")
    
    if train_forms is None or wr_dict is None:
        raise ValueError("Need train_forms and wr_dict")
    
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
    
    # Create lists for each split
    splits = {
        'train': (train_forms, './unhd_data/unhd_train_val_fixed.txt'),
        'val': (val_forms, './unhd_data/unhd_val_fixed.txt'),
        'test': (test_forms, './unhd_data/unhd_test_fixed.txt')
    }
    
    os.makedirs('./unhd_data', exist_ok=True)
    
    for split_name, (form_list, output_file) in splits.items():
        print(f"\nProcessing {split_name} split...")
        lines = []
        skipped = 0
        
        for filename in tqdm(all_files, desc=f"Processing {split_name}"):
            # Extract form name (WWW_LLL from WWW_LLL_V.png)
            name = filename.replace('.png', '')
            parts = name.split('_')
            if len(parts) < 3:
                skipped += 1
                continue
            
            form_name = '_'.join(parts[:-1])  # e.g., 001_001
            
            # Check if this form is in current split
            if form_name not in form_list:
                continue
            
            # Paths
            img_path = os.path.join(data_dir, filename)
            gt_path = os.path.join(data_dir, name + '.gt.txt')
            
            # Check if ground truth exists
            if not os.path.exists(gt_path):
                skipped += 1
                continue
            
            # Read transcription
            try:
                with open(gt_path, 'r', encoding='utf-8') as f:
                    transcr = f.read().strip()
                
                # Remove BOM if present
                if transcr.startswith('\ufeff'):
                    transcr = transcr[1:]
                
                # Skip empty transcriptions
                if not transcr:
                    skipped += 1
                    continue
                
            except Exception as e:
                print(f"Error reading {gt_path}: {e}")
                skipped += 1
                continue
            
            # Get writer ID
            writer_name = parts[0]  # First part (e.g., '001')
            writer_id = wr_dict.get(writer_name, -1)
            
            if writer_id == -1:
                skipped += 1
                continue
            
            # Format: img_path,writer_id,transcription
            # Use comma as separator, but preserve commas in transcription
            lines.append(f"{img_path},{writer_id},{transcr}")
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')
        
        print(f"✓ Created {output_file}")
        print(f"  Valid samples: {len(lines)}")
        print(f"  Skipped: {skipped}")
    
    return True


def verify_setup(data_dir='./unhd_data/UNHD-Complete-Data'):
    """Verify that all files are created correctly"""
    
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    checks = []
    
    # Check split files
    split_files = [
        './unhd_split/train_val.uttlist',
        './unhd_split/validation.uttlist',
        './unhd_split/test.uttlist'
    ]
    
    for f in split_files:
        exists = os.path.exists(f)
        checks.append(exists)
        status = "✓" if exists else "✗"
        print(f"{status} {f}")
        if exists:
            with open(f, 'r') as file:
                count = len(file.readlines())
                print(f"   → {count} forms")
    
    # Check writer dict
    wr_file = './writers_dict.json'
    exists = os.path.exists(wr_file)
    checks.append(exists)
    status = "✓" if exists else "✗"
    print(f"\n{status} {wr_file}")
    if exists:
        with open(wr_file, 'r') as f:
            wr_dict = json.load(f)
            print(f"   → {len(wr_dict)} writers")
    
    # Check data lists
    data_files = [
        './unhd_data/unhd_train_val_fixed.txt',
        './unhd_data/unhd_val_fixed.txt',
        './unhd_data/unhd_test_fixed.txt'
    ]
    
    print()
    for f in data_files:
        exists = os.path.exists(f)
        checks.append(exists)
        status = "✓" if exists else "✗"
        print(f"{status} {f}")
        if exists:
            with open(f, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                count = len(lines)
                print(f"   → {count} samples")
                
                # Show sample
                if count > 0:
                    sample = lines[0].strip().split(',', 2)
                    print(f"   → Sample: writer_id={sample[1]}, text={sample[2][:30]}...")
    
    # Overall result
    print("\n" + "="*60)
    if all(checks):
        print("✓ ALL CHECKS PASSED - Setup complete!")
        print("\nYou can now run:")
        print("  1. python style_encoder_train_unhd.py")
        print("  2. python train_unhd.py")
    else:
        print("✗ SOME CHECKS FAILED - Please review errors above")
    print("="*60)


def show_dataset_statistics(data_dir='./unhd_data/UNHD-Complete-Data'):
    """Show statistics about the dataset"""
    
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    # Load data lists
    splits = {
        'Train': './unhd_data/unhd_train_val_fixed.txt',
        'Val': './unhd_data/unhd_val_fixed.txt',
        'Test': './unhd_data/unhd_test_fixed.txt'
    }
    
    total_samples = 0
    char_set = set()
    transcr_lengths = []
    
    for split_name, filepath in splits.items():
        if not os.path.exists(filepath):
            continue
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            num_samples = len(lines)
            total_samples += num_samples
            
            print(f"\n{split_name} Split:")
            print(f"  Samples: {num_samples}")
            
            # Analyze first 1000 samples
            for line in lines[:1000]:
                parts = line.strip().split(',', 2)
                if len(parts) == 3:
                    transcr = parts[2]
                    char_set.update(transcr)
                    transcr_lengths.append(len(transcr))
    
    print(f"\nOverall:")
    print(f"  Total samples: {total_samples}")
    print(f"  Unique characters: {len(char_set)}")
    
    if transcr_lengths:
        print(f"  Avg transcription length: {sum(transcr_lengths)/len(transcr_lengths):.1f} chars")
        print(f"  Min length: {min(transcr_lengths)}")
        print(f"  Max length: {max(transcr_lengths)}")
    
    print(f"\nCharacter set (first 50):")
    print(''.join(sorted(list(char_set))[:50]))
    
    print("="*60)


def main():
    """Main preprocessing pipeline"""
    
    print("="*60)
    print("UNHD Dataset Preprocessing for DiffusionPen")
    print("="*60)
    
    # Configuration
    data_dir = './unhd_data/UNHD-Complete-Data'
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"\n✗ Error: Data directory not found: {data_dir}")
        print("Please update the path to your UNHD-Complete-Data folder")
        return
    
    # Run preprocessing steps
    train_forms, val_forms, test_forms = create_splits(data_dir)
    wr_dict = create_writer_dict(data_dir)
    create_data_lists(data_dir, train_forms, val_forms, test_forms, wr_dict)
    
    # Verify setup
    verify_setup(data_dir)
    
    # Show statistics
    show_dataset_statistics(data_dir)
    
    print("\n✓ Preprocessing complete!")
    print("\nNext steps:")
    print("1. Train style encoder:")
    print("   python style_encoder_train_unhd.py --epochs 50 --batch_size 64")
    print("\n2. Train DiffusionPen:")
    print("   python train_unhd.py --epochs 1000 --batch_size 64")


if __name__ == '__main__':
    main()