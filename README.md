# dvrk_auto_path_finder

To run the simulation:
1. Use Coppelia scene FinalCoppeliaScene.ttt and select start simulation
2. If you want to run the disparity map model, delte disparity.npy from `demo_output` and set the `restore_ckpt `and `output_directory paths` to your local paths
3. Run the following code
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python image_to_coordinates.py
```
4. Enjoy watching the robot do all sorts of strange things!

##Folder navigation
1. coppelia_elements contains iterations fo the Coppelia scenes
2. demo_output contain uotput of the raftstereo model
3. image_pngs contains the saved images from coppelia scene
4. training_images contains images that could be used for depth training if we had time
5. raftstereo contains teh depth model
6. zmqRemoteApi is the API to communicate with Coppelia
7. The main code is in image_to_coordinates.py
