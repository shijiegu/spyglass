import matplotlib.animation as animation
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sortingview.views as vv
import xarray as xr
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from ripple_detection import get_multiunit_population_firing_rate
from tqdm.auto import tqdm

from spyglass.decoding.v0.visualization_1D_view import create_1D_decode_view
from spyglass.decoding.v0.visualization_2D_view import create_2D_decode_view
from spyglass.utils import logger

import numpy as np
import matplotlib.pyplot as plt
from spyglass.shijiegu.parallel_video_writer import create_parallel_video, VideoConfig
import pickle

def setup_figure():
    # return figure object and a dictionary of axes
    plt.style.use('dark_background')
    
    # Set up plots
    fig, axes = plt.subplots(
            3,
            1,
            figsize=(6.5, 7),
            gridspec_kw={"height_ratios": [5, 1, 1]},
            constrained_layout=False,
    )

    axes[0].tick_params(colors="white", which="both")
    axes[0].spines["bottom"].set_color("white")
    axes[0].spines["left"].set_color("white")



    axes[0].set_xlabel("")
    axes[0].set_ylabel("")
    
    axes[0].spines["top"].set_color("black")
    axes[0].spines["right"].set_color("black")
    
    fontprops = fm.FontProperties(size=16)
    scalebar = AnchoredSizeBar(
            axes[0].transData,
            20,
            "20 cm",
            "lower right",
            pad=0.1,
            color="white",
            frameon=False,
            size_vertical=1,
            fontproperties=fontprops,
    )

    axes[0].add_artist(scalebar)
    axes[0].axis("off")
    axes[0].invert_yaxis()

    
    axes[1].set_ylim((-3, 3))
    #axes[1].set_ylim((0.0, 1))
    
    #axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Multiunit\n[spikes/s]")
    axes[1].set_facecolor("black")
    axes[1].spines["top"].set_color("black")
    axes[1].spines["right"].set_color("black")
        

    axes[2].set_ylim((0.0, 1))
    
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Decode\nContinuous\nstate probability")
    axes[2].set_facecolor("black")
    axes[2].spines["top"].set_color("black")
    axes[2].spines["right"].set_color("black")
    
    print("finished setting up")
    
    return fig, {'main': axes[0], 'MUA': axes[1], 'state': axes[2]}

            
def render_frame(fig, axes, time_ind, data):
    
    print(f"starting to render frame {time_ind}")
    position_info = data["position_info"]
    position = data["position"]
    position_name = data["position_name"]
    posterior_np = data["posterior"]
    sampling_frequency = data["sampling_frequency"]
    direction = data["direction"]
    map_position = data["map_position"]
    rate = data["rate"]
    state_posterior = data["state_posterior"]
    posterior_coords = data["posterior_coords"]
    
    
    
    start_ind = max(0, time_ind - 5)
    time_slice = slice(start_ind, time_ind)
    
    axes["main"].clear()
    
    # animal location
    position_dot = axes["main"].scatter(
            position[time_ind][0],
            position[time_ind][1],
            s=80,
            zorder=102,
            color="magenta",
            label="actual position",
            #animated=True,
        )
    
    # animal head direction
    r = 10
    (position_line,) = axes["main"].plot(
            [
            position[time_ind, 0],
            position[time_ind, 0] + r * np.cos(direction[time_ind])
            ],
            [
            position[time_ind, 1],
            position[time_ind, 1] + r * np.sin(direction[time_ind]),
            ],
            color="magenta", linewidth=5, #animated=True
        )
    
    
    
    
    (map_line,) = axes["main"].plot(map_position[time_slice, 0],
                                    map_position[time_slice, 1], "green", linewidth=3)
    
    # 2D posterior
    posterior = xr.DataArray(posterior_np, coords = posterior_coords, dims=("time", "x_position","y_position"))
    posterior.isel(time=time_ind).values.ravel(order="F")
    posterior.isel(time=time_ind).plot(
            x="x_position",
            y="y_position",
            vmin=0.0,
            vmax=0.06,
            ax=axes["main"],
            add_colorbar=False,
        )
    
    t0 = np.array(posterior.time)[0]
    t_now = posterior.isel(time=time_ind).time.values - t0
    axes["main"].set_title(
                f"time elapsed: {t_now:0.2f} s"
            )
    
    # MUA line
    window_size = 501
    window_ind = np.arange(window_size) - window_size // 2
    indices = window_ind + time_ind + (window_size // 2)
    # middle_ind = indices[int(window_size/2)]
    
    axes["MUA"].clear()
    (multiunit_firing_line,) = axes["MUA"].plot(
            window_ind / sampling_frequency,
            rate.iloc[indices],
            color="white", linewidth=2, clip_on=False
        )
    
    axes["MUA"].set_xlim(
            (
                window_ind[0] / sampling_frequency,
                window_ind[-1] / sampling_frequency,
            )
    )
    axes["MUA"].set_ylim((-3, 3))
    axes['MUA'].set_ylabel("MUA (zscore)")
    
    
    axes['main'].set_xlim(
        107, #position_info[position_name[0]].min() - 10,
        335 #position_info[position_name[0]].max() + 30,
    )
    
    axes['main'].set_ylim(
        22, #position_info[position_name[1]].min() - 10,
        247#position_info[position_name[1]].max() + 10,
    )
    
    #axes['main'].add_artist(scalebar)
    axes['main'].axis("off")
    axes['main'].invert_yaxis()

    # decode state
    state_posterior_subset = np.array(state_posterior.iloc[indices].causal_posterior)

    axes["state"].clear()
    (decode_state_line,) = axes['state'].plot(
            window_ind / sampling_frequency,
            state_posterior_subset,
            color="white", linewidth=2, clip_on=False
    )
    # current_decode_state = state_posterior_subset[int(middle_ind)]
    # if current_decode_state > 0.5:
        # max decode position
    map_dot = axes["main"].scatter(
                map_position[time_ind, 0],
                map_position[time_ind, 1],
                s=80,
                zorder=102,
                color="green",
                label="decoded position",
        )
        

    axes['state'].set_ylim((0.0, 1))
    axes['state'].set_xlim(
            (
                window_ind[0] / sampling_frequency,
                window_ind[-1] / sampling_frequency,
            )
    )
        
    axes['state'].set_xlabel("Time [s]")
    axes['state'].set_ylabel("Decode\nContinuous\nstate probability")
    axes['state'].set_facecolor("black")
    axes['state'].spines["top"].set_color("black")
    axes['state'].spines["right"].set_color("black")
    
    axes['state'].set_xlim(
            (
                window_ind[0] / sampling_frequency,
                window_ind[-1] / sampling_frequency,
            )
    )
    axes['main'].axvline(0, color='white', linestyle='--', linewidth=1)
    axes['state'].axvline(0, color='white', linestyle='--', linewidth=1)

def make_single_environment_movie(
    time_slice,
    environment2D,#classifier,
    results,
    position_info,
    marks,
    movie_name="video_name.mp4",
    sampling_frequency=500,
    video_slowdown=8,
    position_name=["head_position_x", "head_position_y"],
    direction_name="head_orientation",
    vmax=0.06,
    max_workers = 4,
):
    """Generate a movie of the decoding results for a single environment."""
    if marks is not None:
        if marks.ndim > 2:
            multiunit_spikes = (np.any(~np.isnan(marks), axis=1)).astype(float)
        else:
            multiunit_spikes = np.asarray(marks, dtype=float)
        multiunit_firing_rate = pd.DataFrame(
            get_multiunit_population_firing_rate(
                multiunit_spikes, sampling_frequency
            ),
            index=position_info.index,
            columns=["firing_rate"],
        )
        ## Get zscore
        # Calculate mean and standard deviation
        multiunit_firing_rate_subset = multiunit_firing_rate #multiunit_firing_rate[position_info.head_speed >= 4]
        mean_val = multiunit_firing_rate_subset['firing_rate'].mean()
        std_val = multiunit_firing_rate_subset['firing_rate'].std() # By default, ddof=1 for sample standard deviation

        # Calculate z-score
        multiunit_firing_rate['zscore'] = (multiunit_firing_rate['firing_rate'] - mean_val) / std_val
        ##

    # Set up formatting for the movie files
    #Writer = animation.writers["ffmpeg"]
    fps = sampling_frequency // video_slowdown
    #writer = Writer(fps=fps, bitrate=-1)

    # Set up data
    #is_track_interior = classifier.environments[0].is_track_interior
    is_track_interior = environment2D.is_track_interior_
    posterior = (
        results.causal_posterior.isel(time=time_slice)
        .sum("state")
        .where(is_track_interior)
    )

    posterior_np = np.array(posterior)
    coords = posterior.coords
    posterior_coords = {"x_position": np.array(coords["x_position"]), "y_position": np.array(coords["y_position"]), "time":coords["time"]}

    
    map_position_ind = posterior.argmax(["x_position", "y_position"])
    map_position = np.stack(
        (
            posterior.x_position[map_position_ind["x_position"]],
            posterior.y_position[map_position_ind["y_position"]],
        ),
        axis=1,
    )

    position = np.asarray(position_info.iloc[time_slice][position_name])
    direction = np.asarray(position_info.iloc[time_slice][direction_name])

    window_size = 501

    window_ind = np.arange(window_size) - window_size // 2
    wider_slice = slice(
                time_slice.start + window_ind[0], time_slice.stop + window_ind[-1]
            )
    if marks is not None:
        rate = multiunit_firing_rate.iloc[
            wider_slice
        ].zscore
        
        
    state_posterior = results.isel(time=wider_slice).causal_posterior.sum(["x_position","y_position"])
    state_posterior = state_posterior.sel(state = "Continuous").to_dataframe()
    
    n_frames = posterior.shape[0]
    
    frame_data = {"position_info":position_info,
            "position":position,
            "position_name":position_name,
            "posterior":posterior_np,
            "posterior_coords":posterior_coords,
            "sampling_frequency": sampling_frequency,
            "direction": direction,
            "map_position":map_position,
            "rate":rate,
            "state_posterior":state_posterior,
            #"posterior_time":state_posterior_time,
            }
    
    if movie_name is None:
        movie_name = "test.mp4"
        
    config = VideoConfig(fps=fps, dpi=200, max_workers = max_workers)
    create_parallel_video(
        n_frames=n_frames,
        output_path=movie_name,
        render_frame_func=render_frame,
        setup_figure_func=setup_figure,
        frame_data=frame_data,
        config=config
        )
    
    return 1


