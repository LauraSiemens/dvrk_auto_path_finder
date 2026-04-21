# dvrk_auto_path_finder

To run the simulation:
1. Use Coppelia scene **IDK WHICH ONE** and select start simulation
2. If you want to run the disparity map model, delte disparity.npy from `demo_output` and set the `restore_ckpt `and `output_directory paths` to your local paths
3. Run the following code
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python image_to_coordinates.py
```
4. Enjoy watching the robot do all sorts of strange things!