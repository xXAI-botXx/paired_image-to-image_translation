import os
import shutil
import argparse


# Adjust these variables:
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--evaluation_path", type=str, required=True)
parser.add_argument("--target_path", type=str, required=True)
args = parser.parse_args()

name = args.name
evaluation_path = args.evaluation_path
target_path = args.target_path




def get_building_name(full_name):
    """
    buildings_39_real_B.png -> buildings_39.png
    buildings_39_fake_B.png -> buildings_39.png
    """
    splitted = full_name.split('_')
    return f"{splitted[0]}_{splitted[1]}.png"

os.makedirs(target_path, exist_ok=True)

target_real_path = f"{target_path}/real"
os.makedirs(target_real_path, exist_ok=True)

target_pred_path = f"{target_path}/pred"
os.makedirs(target_pred_path, exist_ok=True)

for cur_file_name in os.listdir(evaluation_path):
    cur_file = os.path.join(evaluation_path, cur_file_name)
    if os.path.isfile(cur_file):
        if "real_B" in cur_file_name:
            shutil.copy(cur_file, 
                        os.path.join(target_real_path, get_building_name(cur_file_name)))
            print(f"[info] copied real from '{cur_file}' to '{os.path.join(target_real_path, get_building_name(cur_file_name))}'")
        elif "fake_B" in cur_file_name:
            shutil.copy(cur_file, 
                        os.path.join(target_pred_path, get_building_name(cur_file_name)))
            print(f"[info] copied pred from '{cur_file}' to '{os.path.join(target_pred_path, get_building_name(cur_file_name))}'")
