import os
from PIL import Image

folder = "./unhd_data/UNHD-Complete-Data"
for f in os.listdir(folder):
    if not f.lower().endswith(('.jpg', '.jpeg', '.png')):  # check only images
        continue
    path = os.path.join(folder, f)
    try:
        img = Image.open(path)
        img.verify()  # verify image integrity
    except Exception as e:
        print(f"Bad image: {f}, removing it")
        #os.remove(path)  # or move to a separate folder
