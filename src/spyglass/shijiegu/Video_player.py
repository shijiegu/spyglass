import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pyarrow as pa

R = 10 # position marker size
class VideoPlayer:
    def __init__(self, root, video_path, pos_path):
        self.root = root
        self.root.title("Video Player")

        self.video_frame = tk.Frame(root)
        self.video_frame.pack()

        self.canvas = tk.Canvas(self.video_frame)
        self.canvas.pack()
        
        self.root.bind("<Key>", self.on_key_press)
        
        self.label = tk.Label(root, text="PAUSE/PLAY: Space, PREV: Up or (p), NEXT: Down or (n)", font=("Helvetica", 20, "bold"))
        self.label.pack(padx=10, expand=True)
        
        self.save_button = tk.Button(root, text="Save labels (s)", command=self.save_labels, font=("Helvetica", 20, "bold"))
        self.save_button.pack(padx=10, expand=True)

        # self.play_button = tk.Button(root, text="Play", command=self.play_video, font=("Helvetica", 20, "bold"))
        # self.play_button.pack(padx=10, expand=True)

        # self.pause_button = tk.Button(root, text="Pause", command=self.pause_video, font=("Helvetica", 20, "bold"))
        # self.pause_button.pack(padx=10, expand=True)
        
        # self.next_button = tk.Button(root, text="Next Frame", command=self.advance_one_frame, font=("Helvetica", 20, "bold"))
        # self.next_button.pack(padx=10, expand=True)
        
        # self.prev_button = tk.Button(root, text="Prev Frame", command=self.back_one_frame, font=("Helvetica", 20, "bold"))
        # self.prev_button.pack(padx=10, expand=True)
        
        self.remove_button = tk.Button(root, text="Remove position", command=self.remove_pos, font=("Helvetica", 20, "bold"))
        self.remove_button.pack(padx=10, expand=True)
        
        self.add_button = tk.Button(root, text="Add position", command=self.add_pos, font=("Helvetica", 20, "bold"))
        self.add_button.pack(padx=10, expand=True)

        self.stop_button = tk.Button(root, text="Stop", command=self.stop_video, font=("Helvetica", 20, "bold"))
        self.stop_button.pack(padx=10, expand=True)

        self.video_path = video_path #filedialog.askopenfilename(title="Select Video File", filetypes=[("Video files", "*.mp4;*.avi")])
        self.pos_path = pos_path
        self.pos_data = pd.read_parquet(pos_path)
        self.numFrames = np.array(self.pos_data.video_frame_ind)[-1] + 1
        self.keep_idx = [True for i in np.arange(len(self.pos_data))]
        #self.meters_per_pixel = 1#float(meters_per_pixel)# * 100
        # Create a Label widget with text
        self.label = tk.Label(root, text="Keep Label: True", font=("Helvetica", 20, "bold"))
        self.label.pack(padx=10, expand=True)
        
        self.cap = cv2.VideoCapture(self.video_path)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        self.xloc_f = None
        self.yloc_f = None
        self.xloc_b = None
        self.yloc_b = None
        
        print("width", self.width)
        print("height", self.height)

        self.canvas.config(width=self.width, height=self.height + 30)

        self.paused = True
        self.load_one_frame = False
        self.playing = False # state variable, keeping track of if the video is playing at the moment
        self.current_frame_idx = -1
        self.max_frame_idx = -1
        self.frames = []
        self.frame_idx = []
        
    def on_left_mouse_click(self,event):
        """
        Callback function that handles the mouse click event.
        The 'event' object contains details like x and y coordinates.
        """
        print(f"Left mouse clicked at coordinates: x={event.x}, y={event.y}")
        #self.xloc_f = event.x
        #self.yloc_f = event.y
        if self.allow_inserting:
            print("Updated front LED location")
            self.pos_data.loc[self.pos_data.video_frame_ind == self.current_frame_idx,"xloc2"] = event.x
            self.pos_data.loc[self.pos_data.video_frame_ind == self.current_frame_idx,"yloc2"] = event.y
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.plot_position_one_frame()
        
    def on_right_mouse_click(self,event):
        """
        Callback function that handles the mouse click event.
        The 'event' object contains details like x and y coordinates.
        """
        print(f"Right mouse clicked at coordinates: x={event.x}, y={event.y}")
        #self.xloc_b = 
        #self.yloc_b = event.y
        if self.allow_inserting:
            print("Updated front LED location")
            self.pos_data.loc[self.pos_data.video_frame_ind == self.current_frame_idx,"xloc"] = event.x
            self.pos_data.loc[self.pos_data.video_frame_ind == self.current_frame_idx,"yloc"] = event.y
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.plot_position_one_frame()
        
    def on_key_press(self, event):
        """Callback function executed when a key is pressed."""
        # event.char contains the character of the key pressed
        if event.char.lower() == "n" or event.keysym == "Down":
            ##print(f"Key pressed: {event.char!r}")
            self.advance_one_frame()
        elif event.char.lower() == "p" or event.keysym == "Up":
            ##print(f"Key pressed: {event.char!r}")
            self.back_one_frame()
        elif event.char.lower() == "s":
            ##print(f"Key pressed: {event.char!r}")
            self.save_labels()
        elif event.char.lower() == "r":
            ##print(f"Key pressed: {event.char!r}")
            self.remove_pos()
        elif event.char.lower() == "i":
            ##print(f"Key pressed: {event.char!r}")
            self.insert_labels()
        elif event.char.lower() == "q" or event.keysym == "Escape":
            ##print(f"Key pressed: {event.char!r}")
            self.quit_insert_labels()
        elif event.keysym == "space":
            ##print(f"Key pressed: {event.char!r}")
            if self.paused:
                print("Playing video")
                self.play_video()
            else:
                print("Paused video")
                self.pause_video()
        # For special keys like 'Enter', 'Shift', etc., event.char might be empty.
        # event.keysym can be used for the key's symbol name.
        
    def draw_current_frame(self):
        # draw current frame, without advancing time
        ret = True
        if self.current_frame_idx == self.max_frame_idx:
            frame = self.frames[-1]
        elif self.current_frame_idx < self.max_frame_idx:
            ret = True
            frame = self.frames[self.current_frame_idx - self.max_frame_idx]
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.plot_position_one_frame()
            self.label.config(text=f"Keep Label: {self.keep_idx[self.current_frame_idx]}")
        print("AFTER RENDERING: self.current_frame_idx, self.max_frame_idx")
        print(self.current_frame_idx, self.max_frame_idx)
    
    def update_one_frame(self):
        ret = False
        print("self.current_frame_idx, self.max_frame_idx")
        print(self.current_frame_idx, self.max_frame_idx)
        if self.current_frame_idx + 1 >= self.numFrames:
            print("END OF VIDEO")
            return None
        if self.paused:
            return None
        if self.current_frame_idx == self.max_frame_idx:
            ret, frame = self.cap.read()
            self.frames.append(frame)
            self.frame_idx.append(self.current_frame_idx)
            self.max_frame_idx += 1
        elif self.current_frame_idx < self.max_frame_idx:
            ret = True
            frame = self.frames[self.current_frame_idx - self.max_frame_idx]
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.current_frame_idx += 1 #the renderred image that you see now is current_frame_idx
            self.plot_position_one_frame()
            self.label.config(text=f"Keep Label: {self.keep_idx[self.current_frame_idx]}")
            
            
        print("AFTER RENDERING: self.current_frame_idx, self.max_frame_idx")
        print(self.current_frame_idx, self.max_frame_idx)
    
    
    def update(self):
        self.update_one_frame()
        if self.playing and not self.paused:
            self.root.after(2, self.update)

    def play_video(self):
        if not self.playing:
            print("playing video")
            self.load_one_frame = False
            self.paused = False
            self.playing = True
            self.update()

    def pause_video(self):
        self.paused = True
        self.playing = False
        
    def advance_one_frame(self):
        self.load_one_frame = True
        self.paused = False
        self.playing = False
        self.update_one_frame()
    
    def back_one_frame(self):
        self.load_one_frame = False
        self.paused = False
        self.playing = False
        self.current_frame_idx -= 2
        self.update_one_frame()
        
    def remove_pos(self):
        self.keep_idx[self.current_frame_idx] = False
        print("Removed position for this frame")
        self.draw_current_frame()
    
    def add_pos(self):
        self.keep_idx[self.current_frame_idx] = True
        print("Added position for this frame")
        self.draw_current_frame()
        
    def save_labels(self):
        outputpath = self.pos_path.split(".parquet")[0]+"_reviewed"+".parquet"
        final_pos_df = self.pos_data[self.keep_idx]
        table = pa.Table.from_pandas(final_pos_df)
        pq.write_table(table, outputpath)
        print("Label saved.")
        
    def insert_labels(self):
        self.allow_inserting = True
        print("inserting label.")
        self.canvas.bind("<Button-1>", self.on_left_mouse_click)
        self.canvas.bind("<Button-3>", self.on_right_mouse_click)


    def quit_insert_labels(self):
        print("NOT inserting label now.")
        self.allow_inserting = False
        self.xloc_f = None
        self.xloc_b = None
        
    def plot_position_one_frame(self):
        ind = self.current_frame_idx

        if ind < 0:
            return None
        
        data = self.pos_data[self.pos_data.video_frame_ind == ind]
        if len(data) == 0:
            return None
        print("data", data)
        
        if not self.keep_idx[ind]:
            return None
        (x, y) = (int(data.xloc.iloc[0]), int(data.yloc.iloc[0]))

        self.canvas.create_oval(x-R, y-R, x+R, y+R, fill='red', outline='red')
             
        (x2, y2) = (int(data.xloc2.iloc[0]), int(data.yloc2.iloc[0]))

        self.canvas.create_oval(x2-R, y2-R, x2+R, y2+R, fill='green', outline='green') 

    def stop_video(self):
        self.cap.release()
        self.root.destroy()

    

if __name__ == "__main__":
    # 1. Create the parser
    parser = argparse.ArgumentParser(description='Process a file with optional verbosity.')

    # 2. Add arguments
    # Positional argument (required)
    parser.add_argument('video_path', help='The path to the file that needs processing.')
    #parser.add_argument('pos_path', help='The path to the position file that needs processing.')
    args = parser.parse_args()
    
    root = tk.Tk()
    video_path = args.video_path + "_rawposition.mp4"
    pos_path = args.video_path + ".parquet"
    player = VideoPlayer(root, video_path, pos_path)
    root.mainloop()
