## Prepare Dataset
We collect datasets for **single**, **unknown**, and **composite** degradations. Each dataset is organized in JSON format. Please download the datasets into the corresponding folders, and then use the following command to generate the data list:
```
cd dataset
python download.py
bash decomp_all.sh
python scripts/0_generate_list.py
python scripts/1_generate_iqa_brief.py
```
All datasets can be found on Hugging Face at "AaronCIH/PIR_tar".

### Data Structures:
```
DATA_ROOT/
    ├── $task_name/
    │   ├── $dataset/
    │   │   ├── image_folder/
    │   │   ├── metas/
```
DATA_ROOT is the path containing the counting datasets; Each `dataset/metas` should include: a `subset.list` specifying the image list and a `subset.json` containing the formatted metadata.
