# How To generate the Dataset, train and execute ACT

## Record new episodes
In `liralab_kinova` ros package:
- Uncomment ```REGISTER NEW EPISODES``` code.
- Comment ```RUN ACT``` code.
- ```colcon build```
- Run ```ros2 run liralab_kinova liralab_kinova EPISODE_NAME```.

The new episode is saved in ```ROS/kinova_ws``` ros workspace.

## Generate the Dataset and Train ACT 
Follow the following steps in sequence.
### 0. Activate venv
`source .venv/bin/activate`
### 1. Create the Masks
Run the script `liralab_mask_creator.py` on the episodes folder you want to segment.  
This will generate the corresponding mask inside the `SAVE_PATH` folder.

`python -m liralab.liralab_mask_creator.py`

### 2. Create the H5 Dataset File
Run the script `liralab_dataset_creator.py` on the episodes folder you want to process.
This will convert the CSV files, images, and masks into a single `.hdf5` file.


### 3. Move All H5 Files
Once all episodes have been processed:

- Move all generated `.hdf5` files into the `dataset_dir` folder.
- The `dataset_dir` path is specified inside the `args` structure in `model.py`.

### 4. ❗ **IMPORTANT** ❗Rename the H5 Files 
Rename all `.hdf5` files using the following format: episode_x.hdf5

Where:
- `x` is an incremental number
- Start from `0`

Example:
- `episode_0.hdf5`
- `episode_1.hdf5`
- `episode_2.hdf5`

### 5. Set the Number of Episodes
Inside the `args` structure in `model.py`, set the `num_episodes` parameter according to the number of available episodes.

Example:
If you have:
- `episode_0.hdf5`
- `episode_1.hdf5`
- `episode_2.hdf5`

Then set:

```python
num_episodes = 3
```


### 6. Set the Dataset Directory
Set the `dataset_dir` parameter to the folder containing all the `.hdf5` files.

### 7. Set the Output Directory
Set the `trained_model_dir` parameter to the folder where you want to save the trained model and results.


### 8. ❗ **IMPORTANT** ❗ Restart the Notebook
If you modify any parameter inside `args`, you must restart the notebook:

`Action_Chunking_Transformer.ipynb`

Otherwise, the changes will not be applied.


### 9. Run the Notebook
Run all the cells inside the notebook.

--- 
## Execute ACT

Ensure that all ```.ckpt``` and ```.pth``` are set correctly in ```liralab_IL_control.py``` dictionary.
Ensure that ```DATASET``` field is set to the correct folder (usually ```'data/liralab/AAA'```)

### 0. Connect US pc and check for device ID:
You can run this handy script to get the camera ID:

``` python -m liralab.liralab_camera_finder ```

Then:

- Comment ```REGISTER NEW EPISODES``` code.
- Uncomment ```RUN ACT``` code.
- ```colcon build```

### 1. Run Low Level Controller
``` ros2 run liralab_kinova liralab_kinova ```
### 2. Run Imitation Learning Controller
Set camera ID and run:

``` python -m liralab.liralab_IL_control ```

### 3. Run
Wait for US camera to pop-up. Close it, guide the end effector on the initial position and press ENTER. Everything will run smoothly and safely.