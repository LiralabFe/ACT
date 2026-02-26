# How To Generate the Dataset and train ACT

Follow the following steps in sequence.

## 1. Create the Masks
Run the script `liralab_mask_creator.py` on the episodes folder you want to segment.  
This will generate the corresponding mask inside the `SAVE_PATH` folder.


## 2. Create the H5 Dataset File
Run the script `liralab_dataset_creator.py` on the episodes folder you want to process.
This will convert the CSV files, images, and masks into a single `.h5` file.


## 3. Move All H5 Files
Once all episodes have been processed:

- Move all generated `.h5` files into the `dataset_dir` folder.
- The `dataset_dir` path is specified inside the `args` structure in `models.py`.

## 4. ❗ **(IMPORTANT)** ❗Rename the H5 Files 
Rename all `.h5` files using the following format: episode_x.h5

Where:
- `x` is an incremental number
- Start from `0`

Example:
- `episode_0.h5`
- `episode_1.h5`
- `episode_2.h5`

## 5. Set the Number of Episodes
Inside the `args` structure in `model.py`, set the `num_episodes` parameter according to the number of available episodes.

Example:
If you have:
- `episode_0.h5`
- `episode_1.h5`
- `episode_2.h5`

Then set:

```python
num_episodes = 3
```


## 6. Set the Dataset Directory
Set the `dataset_dir` parameter to the folder containing all the `.h5` files.

## 7. Set the Output Directory
Set the `trained_model_dir` parameter to the folder where you want to save the trained model and results.


## 8. ❗ **(IMPORTANT)** ❗ Restart the Notebook
If you modify any parameter inside `args`, you must restart the notebook:

`Action_Chunking_Transformer.ipynb`

Otherwise, the changes will not be applied.


## 9. Run the Notebook
Run all the cells inside the notebook.

--- 
# How To run ACT
