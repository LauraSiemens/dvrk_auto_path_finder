import csv
import time
import os
import cv2
import numpy as np
from pynput import keyboard, mouse
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# --- Configuration ---
CSV_FILENAME = 'dvrk_stereo_dataset.csv'
IMAGES_FOLDER = 'captured_images'

STEP_XY = 0.002  # Speed for W/A/S/D
STEP_Z = 0.005   # Speed for Mouse Scroll

MODEL_PATHS = {
    # Vision & Ground Truth Tracking
    'camera_L': '/**/Vision_sensor_left',
    'camera_R': '/**/Vision_sensor_right',
    'ring': '/**/ring',
    'base_wire': '/3D_WireOnly',
    'gripper_tip': '/**/L2_respondable_TOOL1', 
    
    # Teleoperation Targets - CRITICAL: These MUST be "Dummy" objects (crosshair icon)
    'ecm_target': '/**/C_tip',
    'psm1_target': '/**/Junction_for_the_instruments_PSM1', # CHANGE THIS to your actual IK Target Dummy!
    
    # Dual Jaw Joints
    'psm1_jaw_dx': '/**/J3_dx_TOOL1',  
    'psm1_jaw_sx': '/**/J3_sx_TOOL1'   
}
# --- End Configuration ---

class DVRKMasterController:
    def __init__(self):
        self.should_capture = False
        self.image_counter = 0
        
        # Teleop State
        self.active_arm = 'psm1_target'
        self.pressed_keys = set()
        self.dz_input = 0.0
        self.toggle_gripper_flag = False
        self.gripper_closed = False
        
        if not os.path.exists(IMAGES_FOLDER):
            os.makedirs(IMAGES_FOLDER)

        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')
        self.handles = {}
        self._get_handles()
        self._init_csv()

    def _get_handles(self):
        print("\n--- Securing Handles ---")
        for key, path in MODEL_PATHS.items():
            try:
                h = self.sim.getObject(path)
                self.handles[key] = h
                print(f"  [OK] {key} -> {path}")
            except Exception as e:
                print(f"  [ERROR] Could not find {path}: {e}")
                raise SystemExit("Missing critical component. Exiting.")

    def _init_csv(self):
        self.header = [
            'timestamp', 'img_L_path', 'img_R_path',
            'ring_w_x', 'ring_w_y', 'ring_w_z',
            'base_w_x', 'base_w_y', 'base_w_z',
            'grip_w_x', 'grip_w_y', 'grip_w_z',
            'ring_camL_x', 'ring_camL_y', 'ring_camL_z',
            'base_camL_x', 'base_camL_y', 'base_camL_z',
            'grip_camL_x', 'grip_camL_y', 'grip_camL_z',
            'fx_L', 'fy_L', 'cx_L', 'cy_L', 'fx_R', 'fy_R', 'cx_R', 'cy_R'
        ]
        if not os.path.exists(CSV_FILENAME):
            with open(CSV_FILENAME, 'w', newline='') as f:
                csv.writer(f).writerow(self.header)

    def get_intrinsics(self, h):
        angle = self.sim.getObjectFloatParam(h, 1004)
        res_x = self.sim.getObjectInt32Param(h, 1002)
        res_y = self.sim.getObjectInt32Param(h, 1003)
        fx = (res_x / 2.0) / np.tan(angle / 2.0)
        fy = (res_y / 2.0) / np.tan(angle / 2.0)
        return fx, fy, res_x/2.0, res_y/2.0, angle

    def save_vision(self, handle, side):
        raw, res_x, res_y = self.sim.getVisionSensorCharImage(handle)
        img = np.frombuffer(raw, dtype=np.uint8).reshape((res_y, res_x, 3))
        img = cv2.cvtColor(cv2.flip(img, 0), cv2.COLOR_RGB2BGR)
        path = os.path.join(IMAGES_FOLDER, f's_{self.image_counter:04d}_{side}.png')
        cv2.imwrite(path, img)
        return path

    def on_press(self, key):
        try:
            char = key.char.lower()
            if char == 's': self.should_capture = True
            else: self.pressed_keys.add(char)
        except AttributeError:
            if key == keyboard.Key.tab:
                self.active_arm = 'ecm_target' if self.active_arm == 'psm1_target' else 'psm1_target'
                arm_name = "CAMERA (ECM)" if self.active_arm == 'ecm_target' else "GRIPPER (PSM1)"
                print(f"\n>>> Switched control to: {arm_name} <<<")

    def on_release(self, key):
        try:
            char = key.char.lower()
            if char in self.pressed_keys: self.pressed_keys.remove(char)
        except AttributeError: pass

    def on_scroll(self, x, y, dx, dy):
        self.dz_input += dy * STEP_Z

    def on_click(self, x, y, button, pressed):
        if button == mouse.Button.right and pressed:
            self.toggle_gripper_flag = True

    def apply_teleop(self):
        # 1. Handle X/Y/Z Movement
        dx, dy = 0.0, 0.0
        if 'w' in self.pressed_keys: dx += STEP_XY
        if 's' in self.pressed_keys: dx -= STEP_XY
        if 'a' in self.pressed_keys: dy += STEP_XY
        if 'd' in self.pressed_keys: dy -= STEP_XY

        if dx != 0 or dy != 0 or self.dz_input != 0:
            h_target = self.handles[self.active_arm]
            
            # Determine the physical tip associated with the active target for the Tether
            h_tip = self.handles['gripper_tip'] if self.active_arm == 'psm1_target' else self.handles['camera_L']

            # Calculate proposed new position
            target_pos = self.sim.getObjectPosition(h_target, -1)
            target_pos[0] += dx
            target_pos[1] += dy
            target_pos[2] += self.dz_input
            
            # THE KINEMATIC TETHER: Check distance between target and actual physical robot
            tip_pos = self.sim.getObjectPosition(h_tip, -1)
            dist = np.linalg.norm(np.array(target_pos) - np.array(tip_pos))

            # If target gets further than 5cm from the tip, the robot has hit a limit constraint.
            # Snap the target back to the tip to prevent it from flying away and breaking IK.
            if dist > 0.05: 
                self.sim.setObjectPosition(h_target, -1, tip_pos)
            else:
                self.sim.setObjectPosition(h_target, -1, target_pos)
                
            self.dz_input = 0.0 

        # 2. Handle Dual-Jaw Gripper Toggle (Kinematic Override)
        if self.toggle_gripper_flag:
            self.toggle_gripper_flag = False
            self.gripper_closed = not self.gripper_closed
            
            angle_dx = 0.0 if self.gripper_closed else 0.5
            angle_sx = 0.0 if self.gripper_closed else -0.5 
            
            # Changed from setJointTargetPosition to setJointPosition for kinematic forcing
            self.sim.setJointPosition(self.handles['psm1_jaw_dx'], angle_dx)
            self.sim.setJointPosition(self.handles['psm1_jaw_sx'], angle_sx)
            
            state = "CLOSED" if self.gripper_closed else "OPENED"
            print(f"Gripper {state}")

    def run(self):
        kb_listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        m_listener = mouse.Listener(on_scroll=self.on_scroll, on_click=self.on_click)
        kb_listener.start()
        m_listener.start()

        print("\n=== SYSTEM READY ===")
        print("[TAB]        : Switch between PSM1 (Gripper) and ECM (Camera)")
        print("[W/A/S/D]    : Move active arm in X/Y plane")
        print("[Scroll]     : Move active arm in Z axis (Up/Down)")
        print("[Right Click]: Toggle Gripper Open/Close")
        print("[S]          : SNAPSHOT (Save data to CSV)")
        print("[Ctrl+C]     : Exit")
        print("====================\n")

        try:
            while True:
                self.apply_teleop()
                
                if self.should_capture:
                    self.should_capture = False
                    h_camL = self.handles['camera_L']
                    
                    row = {
                        'timestamp': self.sim.getSimulationTime(),
                        'img_L_path': self.save_vision(h_camL, 'L'),
                        'img_R_path': self.save_vision(self.handles['camera_R'], 'R')
                    }
                    
                    for obj_key, prefix in [('ring', 'ring'), ('base_wire', 'base'), ('gripper_tip', 'grip')]:
                        h_obj = self.handles[obj_key]
                        pos_w = self.sim.getObjectPosition(h_obj, -1)
                        pos_c = self.sim.getObjectPosition(h_obj, h_camL)
                        
                        row[f'{prefix}_w_x'], row[f'{prefix}_w_y'], row[f'{prefix}_w_z'] = pos_w
                        row[f'{prefix}_camL_x'], row[f'{prefix}_camL_y'], row[f'{prefix}_camL_z'] = pos_c

                    fxL, fyL, cxL, cyL, angle = self.get_intrinsics(h_camL)
                    fxR, fyR, cxR, cyR = self.get_intrinsics(self.handles['camera_R'])
                    row.update({'fx_L': fxL, 'fy_L': fyL, 'cx_L': cxL, 'cy_L': cyL,
                                'fx_R': fxR, 'fy_R': fyR, 'cx_R': cxR, 'cy_R': cyR})

                    with open(CSV_FILENAME, 'a', newline='') as f:
                        csv.DictWriter(f, fieldnames=self.header).writerow(row)
                    
                    print(f"[*] Snapshot {self.image_counter} Saved! | Active Arm: {self.active_arm}")
                    self.image_counter += 1
                
                time.sleep(0.02)
        except KeyboardInterrupt:
            print("\nShutting down master controller.")

if __name__ == "__main__":
    DVRKMasterController().run()