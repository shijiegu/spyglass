from spyglass.common.common_behav import RawPosition
from spyglass.common.common_behav import VideoFile
import pynwb
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

raw_dir = '/stelmo/nwb/raw'

def load_video_and_timestamps(nwb_copy_file_name, session_name):
    key = {'nwb_file_name': nwb_copy_file_name,'epoch':int(session_name[:2])}

    # video info
    video_info = (VideoFile & key).fetch1()

    # parent underscore version nwb path
    nwb_path = f"{raw_dir}/{video_info['nwb_file_name']}"

    # load video timestamp from parent underscore version nwb
    with pynwb.NWBHDF5IO(path=nwb_path, mode="r") as in_out:
        nwb_file = in_out.read()
        nwb_video = nwb_file.objects[video_info["video_file_object_id"]]
        video_filepath = VideoFile.get_abs_path(
            {"nwb_file_name": key["nwb_file_name"], "epoch": key["epoch"]}
        )
        video_dir = os.path.dirname(video_filepath) + "/"
        video_filename = video_filepath.split(video_dir)[-1]
        meters_per_pixel = nwb_video.device.meters_per_pixel
        timestamps = np.asarray(nwb_video.timestamps)

    # load video
    cap = cv2.VideoCapture(video_dir+video_filename)
    
    return cap, timestamps, meters_per_pixel

def make_mp4(cap, timestamps, t0, t1, outputName = None, spatial_df = None):
    frameToPlot = np.argwhere(np.logical_and(timestamps>=t0,timestamps<=t1)).ravel()
    frame0 = frameToPlot[0]
    frameLast = frameToPlot[-1]

    fps = cap.get(cv2.CAP_PROP_FPS) # Gets the frames per second
    frameSize = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    if spatial_df is not None:
        outputName_suffix = '_rawposition_layered.mp4'
    else:
        outputName_suffix = '_rawposition.mp4'
        
    full_outputName = outputName + outputName_suffix
    
    out = cv2.VideoWriter(full_outputName, fourcc, fps, frameSize)

    for fi in range(len(timestamps)):##len(frameToPlot)):
        t = timestamps[fi]

        ret, frame = cap.read()
        frameBrighter = increase_brightness(frame)
        if fi >= frame0 and fi <= frameLast:
            
            if spatial_df is not None:
                # find rat position in cm to pixel
                pos_ind = np.argwhere(spatial_df.index >= t).ravel()[0]
                head_position_x = int(spatial_df.iloc[pos_ind].xloc) #in pixel
                head_position_y = int(spatial_df.iloc[pos_ind].yloc) #in pixel
            
                # convert rat to pixel
                plt.scatter(head_position_x, head_position_y,color = 'C0')
        
                if np.isnan(head_position_x) or np.isnan(head_position_y):
                    continue
            
                cv2.circle(frameBrighter, (head_position_x, head_position_y), 3, (0, 0, 255), -1)

            out.write(frameBrighter)
        if fi > frameLast:
            break
            
    cap.release()
    out.release()
    return full_outputName

def increase_brightness(img, value=20):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img