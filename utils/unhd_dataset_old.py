import numpy as np
from skimage import io as img_io
from utils.word_dataset import WordLineDataset
from utils.auxilary_functions import image_resize_PIL, centered_PIL
from PIL import Image, ImageOps
import json
import os
import string
import random  # For crops if needed

class UNHDDataset(WordLineDataset):
    def __init__(self, basefolder, subset, segmentation_level, fixed_size, tokenizer, text_encoder, feat_extractor, transforms, args):
        super().__init__(basefolder, subset, segmentation_level, fixed_size, tokenizer, text_encoder, feat_extractor, transforms, args)
        self.setname = 'UNHD'
        self.trainset_file = './unhd_split/train_val.uttlist'
        self.valset_file = './unhd_split/validation.uttlist'
        self.testset_file = './unhd_split/test.uttlist'
        self.word_path = self.basefolder  # UNHD is flat, no subdirs
        self.line_path = self.basefolder  # Same for lines
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.feat_extractor = feat_extractor
        self.args = args
        super().__finalize__()

    # Optional: If you need crops, but unused in IAM code
    # def generate_multiple_crops(img, num_crops=4, crop_size=(200, 50)):
    #     ... (same as in IAM)

    def main_loader(self, subset, segmentation_level) -> list:
        def gather_unhd_info(self, set='train', level='line'):  # Assume line-level for UNHD
            if subset == 'train':
                valid_set = np.loadtxt(self.trainset_file, dtype=str)
            elif subset == 'val':
                valid_set = np.loadtxt(self.valset_file, dtype=str)
            elif subset == 'test':
                valid_set = np.loadtxt(self.testset_file, dtype=str)
            else:
                raise ValueError

            if level != 'line':  # UNHD is lines, so enforce
                raise ValueError("UNHD supports 'line' level only")

            root_path = self.line_path
            print('root_path', root_path)

            dict_path = './writers_dict.json'  # Single dict for all
            with open(dict_path, 'r') as f:
                wr_dict = json.load(f)

            gt = []
            # Iterate over all PNG files in the folder
            all_files = [f for f in os.listdir(root_path) if f.endswith('.png')]
            for i, filename in enumerate(all_files):
                if i % 1000 == 0:
                    print(f'Processing files: [{i}/{len(all_files)} ({100. * i / len(all_files):.0f}%)]')

                name = filename.replace('.png', '')  # e.g., 001_001_1
                form_name = '_'.join(name.split('_')[:-1])  # e.g., 001_001

                if form_name not in valid_set:
                    continue

                img_path = os.path.join(root_path, filename)
                gt_path = os.path.join(root_path, name + '.gt.txt')

                if not os.path.exists(gt_path):
                    continue

                # Read transcription (Urdu as-is, no IAM-specific cleaning)
                with open(gt_path, 'r', encoding='utf-8') as f:
                    transcr = f.read().strip()

                writer_name = name.split('_')[0]  # e.g., '001'
                writer_id = wr_dict.get(writer_name, -1)  # Map to int
                if writer_id == -1:
                    continue

                gt.append((img_path, transcr, writer_id, name))  # Use writer_id (int)

            return gt

        info = gather_unhd_info(self, subset, segmentation_level)
        data = []
        widths = []
        padded_imgs = 0
        padded_data = []
        character_classes = ['!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']  # Keep IAM's, but model will learn Urdu chars from data

        for i, (img_path, transcr, writer_name, name) in enumerate(info):
            if i % 1000 == 0:
                print('imgs: [{}/{} ({:.0f}%)]'.format(i, len(info), 100. * i / len(info)))

            try:
                img = Image.open(img_path).convert('RGB')  # .convert('L') if grayscale needed

                # Skip IAM punctuation check; for Urdu, process all
                (img_width, img_height) = img.size
                # Resize to height 64 keeping aspect ratio
                img = img.resize((int(img_width * 64 / img_height), 64))
                (img_width, img_height) = img.size

                if img_width < 256:
                    outImg = ImageOps.pad(img, size=(256, 64), color="white")
                    img = outImg
                else:
                    # Reduce until width <= 256
                    while img_width > 256:
                        img = image_resize_PIL(img, width=img_width - 20)
                        (img_width, img_height) = img.size
                    img = centered_PIL(img, (64, 256), border_value=255.0)

                # Optional self-padding (commented in IAM; keep commented unless needed)
                # ...

            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue

            data += [(img, transcr, writer_name, img_path)]

        print('len data', len(data))
        return data