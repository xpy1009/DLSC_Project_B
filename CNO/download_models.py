import os

print("This script will download all the trained models required for benchmark problems."
      " You must have 'wget' installed for the downloads to work.")

folder_name = "./"

fnames = ["SELECTED_MODELS.zip"]
for fid, fname in enumerate(fnames):
    print('Downloading file {fname} ({fid+1}/{len(fnames)}):')
    url = "https://zenodo.org/record/7963379/files/" + fname
    cmd = f"wget --directory-prefix {folder_name} {url}"
    os.system(cmd)
