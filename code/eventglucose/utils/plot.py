import pandas as pd
import numpy as np
import os
import matplotlib

# When running large batch experiments, force a non-interactive backend to avoid
# any accidental GUI rendering and to reduce memory usage.
# Enable with: export CIK_NO_PLOT_DISPLAY=1
if os.environ.get("CIK_NO_PLOT_DISPLAY", "0") == "1":
    matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


def plot_task(task, prediction_samples=None, max_sample_traces=10):
    """
    Plots a task's numerical data and returns the figure.

    Parameters:
    -----------
    task: BaseTask
        The task to plot
    prediction_samples: np.ndarray, optional
        Prediction samples with shape (n_samples, time_steps, 1) or (n_samples, time_steps)
    max_sample_traces: int
        Maximum number of prediction sample traces to show

    Returns:
    --------
    fig: matplotlib.figure.Figure
        The figure containing the plot

    """
    # Use 12:5 aspect ratio for consistency
    fig = plt.figure(figsize=(15, 8))

    # Prepare data
    past_timesteps = task.past_time.index
    past_values = task.past_time.to_numpy()[:, -1]
    future_timesteps = task.future_time.index
    future_values = task.future_time.to_numpy()[:, -1]

    # The fill_between method is only ok with pd.DatetimeIndex
    if isinstance(past_timesteps, pd.PeriodIndex):
        past_timesteps = past_timesteps.to_timestamp()
    if isinstance(future_timesteps, pd.PeriodIndex):
        future_timesteps = future_timesteps.to_timestamp()

    timesteps = np.concatenate([past_timesteps, future_timesteps])
    values = np.concatenate([past_values, future_values])

    # Plot the history (blue, thick line)
    plt.plot(
        past_timesteps,
        past_values,
        color="blue",
        linewidth=3,
        alpha=0.8,
        zorder=5,
        label="Historical glucose",
    )

    # Plot the ground truth (orange, thick line)
    plt.plot(
        future_timesteps,
        future_values,
        color="orange",
        linewidth=3,
        alpha=0.8,
        zorder=5,
        label="Future ground truth",
    )

    # Plot prediction samples if provided
    if prediction_samples is not None:
        # Handle different input shapes
        if prediction_samples.ndim == 3:
            prediction_samples = prediction_samples[:, :, 0]  # Remove last dimension

        # Plot individual sample traces (thin gray lines)
        n_traces = min(max_sample_traces, prediction_samples.shape[0])
        if n_traces > 0:
            indices = np.linspace(0, prediction_samples.shape[0] - 1, n_traces).astype(int)
            for i in indices:
                plt.plot(future_timesteps, prediction_samples[i],
                        color='gray', linewidth=0.7, alpha=0.6, zorder=2)

        # Compute and plot prediction statistics
        pred_mean = prediction_samples.mean(axis=0)
        pred_q05, pred_q25 = np.percentile(prediction_samples, [5, 25], axis=0)
        pred_q75, pred_q95 = np.percentile(prediction_samples, [75, 95], axis=0)

        # Confidence bands
        plt.fill_between(future_timesteps, pred_q05, pred_q95, alpha=0.25,
                    color='lightcoral', label='90% prediction band', zorder=1)
        plt.fill_between(future_timesteps, pred_q25, pred_q75, alpha=0.4,
                    color='lightgreen', label='50% prediction band', zorder=1)

        # Prediction mean
        plt.plot(future_timesteps, pred_mean, color='red', linewidth=2.5,
            label='Prediction mean', zorder=4)

    # Add vertical line at prediction boundary
    boundary_time = future_timesteps[0]
    plt.axvline(x=boundary_time, color='red', linestyle='-', alpha=0.8,
               linewidth=2.5, label='Prediction boundary')

    # Add event markers if this is a GlucoseCGMTask with events
    if hasattr(task, 'c_cov'):
        # Get intervention data
        past_cov = task.c_cov['past']
        future_cov = task.c_cov['future']

        # Last 3 columns are diet, med, exercise
        past_diet = past_cov[:, -3]
        past_med = past_cov[:, -2]
        past_ex = past_cov[:, -1]

        future_diet = future_cov[:, -3]
        future_med = future_cov[:, -2]
        future_ex = future_cov[:, -1]

        # Past interventions
        diet_past_idx = np.where(past_diet > 0)[0]
        med_past_idx = np.where(past_med > 0)[0]
        ex_past_idx = np.where(past_ex > 0)[0]

        if len(diet_past_idx) > 0:
            plt.scatter(past_timesteps[diet_past_idx], past_values[diet_past_idx],
                       c='green', s=150, marker='^', alpha=0.7, zorder=6, label='Diet')

        if len(med_past_idx) > 0:
            plt.scatter(past_timesteps[med_past_idx], past_values[med_past_idx],
                       c='red', s=150, marker='o', alpha=0.7, zorder=6, label='Medication')

        if len(ex_past_idx) > 0:
            plt.scatter(past_timesteps[ex_past_idx], past_values[ex_past_idx],
                       c='darkorange', s=150, marker='s', alpha=0.7, zorder=6, label='Exercise')

        # Future interventions
        diet_future_idx = np.where(future_diet > 0)[0]
        med_future_idx = np.where(future_med > 0)[0]
        ex_future_idx = np.where(future_ex > 0)[0]

        # Add labels only if not already in legend from past interventions
        diet_label = None if len(diet_past_idx) > 0 else 'Diet'
        med_label = None if len(med_past_idx) > 0 else 'Medication'
        ex_label = None if len(ex_past_idx) > 0 else 'Exercise'

        if len(diet_future_idx) > 0:
            plt.scatter(future_timesteps[diet_future_idx], future_values[diet_future_idx],
                       c='green', s=150, marker='^', alpha=0.7, zorder=6, label=diet_label)

        if len(med_future_idx) > 0:
            plt.scatter(future_timesteps[med_future_idx], future_values[med_future_idx],
                       c='red', s=150, marker='o', alpha=0.7, zorder=6, label=med_label)

        if len(ex_future_idx) > 0:
            plt.scatter(future_timesteps[ex_future_idx], future_values[ex_future_idx],
                       c='darkorange', s=150, marker='s', alpha=0.7, zorder=6, label=ex_label)

    # For GlucoseCGMTask_withEvent_withLag, add event line and lag visualization
    if hasattr(task, 'selected_event'):
        event_info = task.selected_event
        event_idx = event_info.get('prediction_time_idx', event_info.get('index'))  # Support both old and new
        lag = event_info.get('lag', 0)

        # Get the observation point (cut index)
        obs_point = event_info.get('obs_point', event_idx)

        # The past window is [obs_point - C, obs_point)
        C = len(past_values)

        # Event position relative to observation point
        event_pos_relative = event_idx - (obs_point - C)

        # If event is within our visualization window
        if 0 <= event_pos_relative < len(timesteps):
            event_time = timesteps[event_pos_relative]
            event_value = values[event_pos_relative]

            # Add vertical line at event time
            plt.axvline(x=event_time, color='purple', linestyle='--', alpha=0.7,
                       linewidth=2, label=f'Event ({event_info["type"].replace("5Min", "")})')

            # Add event marker on the glucose curve
            event_color = {'Diet5Min': 'green', 'Med5Min': 'red', 'Exercise5Min': 'darkorange'}
            event_marker = {'Diet5Min': '^', 'Med5Min': 'o', 'Exercise5Min': 's'}

            color = event_color.get(event_info['type'], 'purple')
            marker = event_marker.get(event_info['type'], '*')

            plt.scatter([event_time], [event_value], c=color, s=300,
                       marker=marker, alpha=1.0, zorder=7,
                       edgecolors='black', linewidth=2.5)

            # If there's a lag, show it visually
            if lag != 0:
                # Add arrow showing lag relationship
                if lag > 0:
                    # Event is in the past relative to prediction boundary
                    plt.annotate('', xy=(boundary_time, 50), xytext=(event_time, 50),
                                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5, alpha=0.6))
                    # Calculate midpoint for text placement
                    if isinstance(event_time, pd.Timestamp) and isinstance(boundary_time, pd.Timestamp):
                        mid_time = event_time + (boundary_time - event_time)/2
                    else:
                        mid_time = (event_time + boundary_time) / 2
                    plt.text(mid_time, 60, f'Lag: {abs(lag)*5} min', ha='center', fontsize=9)
                elif lag < 0:
                    # Event is in the future relative to prediction boundary
                    plt.annotate('', xy=(event_time, 50), xytext=(boundary_time, 50),
                                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5, alpha=0.6))
                    # Calculate midpoint for text placement
                    if isinstance(event_time, pd.Timestamp) and isinstance(boundary_time, pd.Timestamp):
                        mid_time = boundary_time + (event_time - boundary_time)/2
                    else:
                        mid_time = (boundary_time + event_time) / 2
                    plt.text(mid_time, 60, f'Lag: {abs(lag)*5} min', ha='center', fontsize=9)

    # Optional: Shade the entire future region in very light gray to indicate the forecast region
    # Commented out for cleaner look - uncomment if desired
    # plt.fill_between(
    #     future_timesteps,
    #     -99999,  # Use the current minimum y-limit
    #     99999,  # Use the current maximum y-limit
    #     facecolor=(0.95, 0.95, 0.95),
    #     interpolate=True,
    #     label="Forecast region",
    #     zorder=0,
    # )

    # Shade RoI
    if type(task.region_of_interest) != type(None):
        if isinstance(task.region_of_interest, list):
            # Convert the list of timesteps to a Pandas Series and find contiguous groups
            roi_series = pd.Series(task.region_of_interest)
            contiguous_regions = []
            start_idx = roi_series.iloc[0]
            for i in range(1, len(roi_series)):
                if roi_series.iloc[i] != roi_series.iloc[i - 1] + 1:
                    contiguous_regions.append(slice(start_idx, roi_series.iloc[i - 1]))
                    start_idx = roi_series.iloc[i]
            contiguous_regions.append(slice(start_idx, roi_series.iloc[-1]))
        elif isinstance(task.region_of_interest, int):
            contiguous_regions = [[task.region_of_interest]]
        else:
            contiguous_regions = [task.region_of_interest]
        for region_index, region in enumerate(contiguous_regions):
            plt.fill_between(
                future_timesteps[region],
                -99999,  # Use the current minimum y-limit
                99999,  # Use the current maximum y-limit
                facecolor=(0, 0.8, 0),
                interpolate=True,
                label="Region of Interest" if region_index == 0 else None,
                zorder=0,
            )

    # Add glucose range thresholds
    plt.axhline(y=70, color='gray', linestyle='--', alpha=0.3)
    plt.axhline(y=180, color='gray', linestyle='--', alpha=0.3)

    # Minor style tweaks
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.01, 1.05), loc='upper left', fontsize=9)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Glucose (mg/dL)', fontsize=12)

    # Enhanced title with predictions info
    title = f'{task.__class__.__name__ if hasattr(task, "__class__") else "Task"}'
    if prediction_samples is not None:
        title += ' - Predictions vs Ground Truth'
    plt.title(title, fontsize=14, fontweight='bold')

    plt.grid(True, alpha=0.2)

    # Set axis limits - MUST be after all plot operations and before tight_layout
    plt.xlim(timesteps[0], timesteps[-1])
    plt.ylim(0, 450)  # Force y-axis to be 0-450

    plt.tight_layout()

    # Ensure y-axis limits are still 0-450 after tight_layout (sometimes tight_layout can change them)
    plt.ylim(0, 450)

    return plt.gcf()




"""
Visualization functions for glucose time series data with intervention events
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from typing import Union, Dict, Optional


def parse_event_dict(event_data: Union[str, dict, None]) -> dict:
    """
    Parse event data from various formats into a dictionary.
    
    Parameters:
    -----------
    event_data : str, dict, or None
        Event data in string representation of dict or actual dict
        
    Returns:
    --------
    dict: Parsed event dictionary with integer keys
    """
    if event_data is None or (isinstance(event_data, str) and event_data.strip() == ''):
        return {}
    
    if isinstance(event_data, str):
        try:
            return ast.literal_eval(event_data)
        except:
            return {}
    
    if isinstance(event_data, dict):
        return event_data
    
    return {}


def visualize_glucose_with_events(
    df: pd.DataFrame,
    row_index: int = 0,
    figsize: tuple = (15, 6),
    show_legend: bool = True
) -> None:
    """
    Visualize a single glucose time series with Diet and Exercise events marked.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The glucose dataframe with columns: target, start, Diet5Min, Exercise5Min
    row_index : int
        Which row of the dataframe to visualize
    figsize : tuple
        Figure size for the plot
    show_legend : bool
        Whether to show the legend
    """
    # Get the specific row
    row = df.iloc[row_index]
    
    # Parse the glucose values
    if isinstance(row['target'], str):
        glucose_values = np.array(ast.literal_eval(row['target']), dtype=float)
    else:
        glucose_values = np.array(row['target'], dtype=float)
    
    # Create time index (5-minute intervals)
    start_time = pd.to_datetime(row['start'])
    time_index = pd.date_range(start=start_time, periods=len(glucose_values), freq='5min')
    
    # Parse event dictionaries
    diet_events = parse_event_dict(row.get('Diet5Min', {}))
    exercise_events = parse_event_dict(row.get('Exercise5Min', {}))
    med_events = parse_event_dict(row.get('Med5Min', {}))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot glucose time series
    ax.plot(time_index, glucose_values, 'b-', linewidth=1, alpha=0.7, label='Glucose (mg/dL)')
    
    # Mark Diet events
    if diet_events:
        diet_indices = [int(k) for k in diet_events.keys() if 0 <= int(k) < len(glucose_values)]
        diet_times = [time_index[i] for i in diet_indices]
        diet_values = [glucose_values[i] for i in diet_indices]
        ax.scatter(diet_times, diet_values, c='green', s=100, marker='^', 
                  alpha=0.8, label=f'Diet ({len(diet_indices)} events)', zorder=5)
    
    # Mark Exercise events
    if exercise_events:
        exercise_indices = [int(k) for k in exercise_events.keys() if 0 <= int(k) < len(glucose_values)]
        exercise_times = [time_index[i] for i in exercise_indices]
        exercise_values = [glucose_values[i] for i in exercise_indices]
        ax.scatter(exercise_times, exercise_values, c='orange', s=100, marker='s', 
                  alpha=0.8, label=f'Exercise ({len(exercise_indices)} events)', zorder=5)
    
    # Mark Medication events
    if med_events:
        med_indices = [int(k) for k in med_events.keys() if 0 <= int(k) < len(glucose_values)]
        med_times = [time_index[i] for i in med_indices]
        med_values = [glucose_values[i] for i in med_indices]
        ax.scatter(med_times, med_values, c='red', s=100, marker='o', 
                  alpha=0.8, label=f'Medication ({len(med_indices)} events)', zorder=5)
    
    # Add horizontal lines for glucose ranges
    ax.axhline(y=70, color='gray', linestyle='--', alpha=0.3, label='Low threshold (70 mg/dL)')
    ax.axhline(y=180, color='gray', linestyle='--', alpha=0.3, label='High threshold (180 mg/dL)')
    
    # Formatting
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Glucose (mg/dL)', fontsize=12)
    
    # Title with metadata
    patient_id = row.get('PatientID', 'Unknown')
    item_id = row.get('item_id', 'Unknown')
    phase = row.get('phase', 'Unknown')
    ax.set_title(f'Glucose Time Series - Patient: {patient_id}, Item: {item_id}, Phase: {phase}', 
                fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add grid
    ax.grid(True, alpha=0.2)
    
    # Set y-axis limits
    ax.set_ylim(0, 450)
    
    # Legend
    if show_legend:
        ax.legend(loc='upper right', fontsize=10)
    
    # Add statistics box
    stats_text = f'Mean: {glucose_values.mean():.1f} mg/dL\n'
    stats_text += f'Std: {glucose_values.std():.1f} mg/dL\n'
    stats_text += f'Min: {glucose_values.min():.1f} mg/dL\n'
    stats_text += f'Max: {glucose_values.max():.1f} mg/dL'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    # plt.show()
    
    # Print event summary
    print(f"\nEvent Summary for Row {row_index}:")
    print(f"- Total glucose readings: {len(glucose_values)}")
    print(f"- Duration: {(time_index[-1] - time_index[0]).days} days, "
          f"{(time_index[-1] - time_index[0]).seconds // 3600} hours")
    print(f"- Diet events: {len(diet_events)}")
    print(f"- Exercise events: {len(exercise_events)}")
    print(f"- Medication events: {len(med_events)}")


def visualize_multiple_series(
    df: pd.DataFrame,
    n_samples: int = 3,
    figsize: tuple = (15, 4),
    random_state: Optional[int] = None
) -> None:
    """
    Visualize multiple glucose time series in subplots.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The glucose dataframe
    n_samples : int
        Number of samples to visualize
    figsize : tuple
        Figure size per subplot
    random_state : int
        Random seed for sample selection
    """
    # Sample rows
    if random_state is not None:
        sample_df = df.sample(n=min(n_samples, len(df)), random_state=random_state)
    else:
        sample_df = df.head(min(n_samples, len(df)))
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(figsize[0], figsize[1] * n_samples))
    
    if n_samples == 1:
        axes = [axes]
    
    for idx, (_, row) in enumerate(sample_df.iterrows()):
        ax = axes[idx]
        
        # Parse glucose values
        if isinstance(row['target'], str):
            glucose_values = np.array(ast.literal_eval(row['target']), dtype=float)
        else:
            glucose_values = np.array(row['target'], dtype=float)
        
        # Create time index
        start_time = pd.to_datetime(row['start'])
        time_index = pd.date_range(start=start_time, periods=len(glucose_values), freq='5min')
        
        # Parse events
        diet_events = parse_event_dict(row.get('Diet5Min', {}))
        exercise_events = parse_event_dict(row.get('Exercise5Min', {}))
        med_events = parse_event_dict(row.get('Med5Min', {}))
        
        # Plot glucose
        ax.plot(time_index, glucose_values, 'b-', linewidth=0.8, alpha=0.7)
        
        # Mark events
        if diet_events:
            diet_indices = [int(k) for k in diet_events.keys() if 0 <= int(k) < len(glucose_values)]
            diet_times = [time_index[i] for i in diet_indices]
            diet_values = [glucose_values[i] for i in diet_indices]
            ax.scatter(diet_times, diet_values, c='green', s=80, marker='^', alpha=0.8)
        
        if exercise_events:
            exercise_indices = [int(k) for k in exercise_events.keys() if 0 <= int(k) < len(glucose_values)]
            exercise_times = [time_index[i] for i in exercise_indices]
            exercise_values = [glucose_values[i] for i in exercise_indices]
            ax.scatter(exercise_times, exercise_values, c='orange', s=80, marker='s', alpha=0.8)
        
        if med_events:
            med_indices = [int(k) for k in med_events.keys() if 0 <= int(k) < len(glucose_values)]
            med_times = [time_index[i] for i in med_indices]
            med_values = [glucose_values[i] for i in med_indices]
            ax.scatter(med_times, med_values, c='red', s=80, marker='o', alpha=0.8)
        
        # Add threshold lines
        ax.axhline(y=70, color='gray', linestyle='--', alpha=0.2)
        ax.axhline(y=180, color='gray', linestyle='--', alpha=0.2)
        
        # Labels
        patient_id = row.get('PatientID', 'Unknown')
        ax.set_title(f'Patient {patient_id} - {len(glucose_values)} readings, '
                    f'Diet: {len(diet_events)}, Exercise: {len(exercise_events)}, Med: {len(med_events)}',
                    fontsize=10)
        ax.set_ylabel('Glucose\n(mg/dL)', fontsize=9)
        ax.grid(True, alpha=0.2)
        
        # Set y-axis limits
        ax.set_ylim(0, 450)
        
        ax.set_xlabel('Time', fontsize=10)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Only show x-label on last subplot
        # if idx == n_samples - 1:
        #     ax.set_xlabel('Time', fontsize=10)
        #     plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        # else:
        #     ax.set_xticklabels([])
    
    plt.suptitle('Glucose Time Series with Intervention Events', fontsize=14, fontweight='bold')
    plt.tight_layout()
    # plt.show()


def plot_event_distribution(df: pd.DataFrame) -> None:
    """
    Plot the distribution of events across all patients.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The glucose dataframe
    """
    diet_counts = []
    exercise_counts = []
    med_counts = []
    
    for _, row in df.iterrows():
        diet_events = parse_event_dict(row.get('Diet5Min', {}))
        exercise_events = parse_event_dict(row.get('Exercise5Min', {}))
        med_events = parse_event_dict(row.get('Med5Min', {}))
        
        diet_counts.append(len(diet_events))
        exercise_counts.append(len(exercise_events))
        med_counts.append(len(med_events))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Diet events histogram
    axes[0].hist(diet_counts, bins=30, color='green', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Number of Diet Events')
    axes[0].set_ylabel('Number of Patients')
    axes[0].set_title(f'Diet Events Distribution\n(Mean: {np.mean(diet_counts):.1f}, '
                     f'Median: {np.median(diet_counts):.0f})')
    axes[0].grid(True, alpha=0.3)
    
    # Exercise events histogram
    axes[1].hist(exercise_counts, bins=30, color='orange', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Number of Exercise Events')
    axes[1].set_ylabel('Number of Patients')
    axes[1].set_title(f'Exercise Events Distribution\n(Mean: {np.mean(exercise_counts):.1f}, '
                     f'Median: {np.median(exercise_counts):.0f})')
    axes[1].grid(True, alpha=0.3)
    
    # Medication events histogram
    axes[2].hist(med_counts, bins=30, color='red', alpha=0.7, edgecolor='black')
    axes[2].set_xlabel('Number of Medication Events')
    axes[2].set_ylabel('Number of Patients')
    axes[2].set_title(f'Medication Events Distribution\n(Mean: {np.mean(med_counts):.1f}, '
                     f'Median: {np.median(med_counts):.0f})')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Distribution of Intervention Events Across All Patients', fontsize=14, fontweight='bold')
    plt.tight_layout()
    # plt.show()
    
    # Print summary statistics
    print("\nEvent Statistics Across All Patients:")
    print(f"Diet Events - Total: {sum(diet_counts)}, "
          f"Patients with events: {sum(1 for x in diet_counts if x > 0)}/{len(df)}")
    print(f"Exercise Events - Total: {sum(exercise_counts)}, "
          f"Patients with events: {sum(1 for x in exercise_counts if x > 0)}/{len(df)}")
    print(f"Medication Events - Total: {sum(med_counts)}, "
          f"Patients with events: {sum(1 for x in med_counts if x > 0)}/{len(df)}")


# Example usage in notebook:
if __name__ == "__main__":
    print("Usage example:")
    print("---------------")
    print("import pandas as pd")
    print("from visualize_glucose_events import visualize_glucose_with_events, visualize_multiple_series, plot_event_distribution")
    print("")
    print("# Load your data")
    print("df = pd.read_pickle(f'{data_folder}/df_lts.pkl')")
    print("")
    print("# Visualize a single series with events")
    print("visualize_glucose_with_events(df, row_index=0)")
    print("")
    print("# Visualize multiple series")
    print("visualize_multiple_series(df, n_samples=3, random_state=42)")
    print("")
    print("# Plot event distribution across all patients")
    print("plot_event_distribution(df)")


# def visualize_task_instance(
#     task,
#     show_event_marker: bool = True,
#     figsize: tuple = (15, 6),
#     show_interventions: bool = True
# ):
#     """
#     Visualize a GlucoseCGMTask or GlucoseCGMTask_withEvent instance.
#
#     Parameters:
#     -----------
#     task : GlucoseCGMTask or GlucoseCGMTask_withEvent
#         The task instance after calling random_instance()
#     show_event_marker : bool
#         Whether to show the selected event marker (for withEvent tasks)
#     figsize : tuple
#         Figure size
#     show_interventions : bool
#         Whether to show intervention markers
#     """
#     # Get glucose data
#     past_glucose = task.past_time['glucose_mg_dl']
#     future_glucose = task.future_time['glucose_mg_dl']
#
#     # Create combined time series for visualization
#     combined_time = pd.concat([past_glucose, future_glucose])
#
#     fig, ax = plt.subplots(figsize=figsize)
#
#     # Plot historical data
#     ax.plot(past_glucose.index, past_glucose.values, 'b-',
#            linewidth=1.5, alpha=0.8, label='Historical glucose')
#
#     # Plot future data (ground truth)
#     ax.plot(future_glucose.index, future_glucose.values, 'g--',
#            linewidth=1.5, alpha=0.8, label='Future glucose (ground truth)')
#
#     # Add vertical line at prediction boundary
#     boundary_time = future_glucose.index[0]
#     ax.axvline(x=boundary_time, color='red', linestyle='-', alpha=0.7,
#               linewidth=2, label='Prediction boundary')
#
#     # Show interventions if available
#     if show_interventions and hasattr(task, 'c_cov'):
#         # Get intervention data
#         past_cov = task.c_cov['past']
#         future_cov = task.c_cov['future']
#
#         # Last 3 columns are diet, med, exercise
#         past_diet = past_cov[:, -3]
#         past_med = past_cov[:, -2]
#         past_ex = past_cov[:, -1]
#
#         future_diet = future_cov[:, -3]
#         future_med = future_cov[:, -2]
#         future_ex = future_cov[:, -1]
#
#         # Plot intervention markers
#         past_times = past_glucose.index
#         future_times = future_glucose.index
#
#         # Past interventions
#         diet_past_idx = np.where(past_diet > 0)[0]
#         med_past_idx = np.where(past_med > 0)[0]
#         ex_past_idx = np.where(past_ex > 0)[0]
#
#         if len(diet_past_idx) > 0:
#             ax.scatter(past_times[diet_past_idx], past_glucose.iloc[diet_past_idx],
#                       c='green', s=80, marker='^', alpha=0.9, zorder=5, label='Diet')
#
#         if len(med_past_idx) > 0:
#             ax.scatter(past_times[med_past_idx], past_glucose.iloc[med_past_idx],
#                       c='red', s=80, marker='o', alpha=0.9, zorder=5, label='Medication')
#
#         if len(ex_past_idx) > 0:
#             ax.scatter(past_times[ex_past_idx], past_glucose.iloc[ex_past_idx],
#                       c='orange', s=80, marker='s', alpha=0.9, zorder=5, label='Exercise')
#
#         # Future interventions
#         diet_future_idx = np.where(future_diet > 0)[0]
#         med_future_idx = np.where(future_med > 0)[0]
#         ex_future_idx = np.where(future_ex > 0)[0]
#
#         if len(diet_future_idx) > 0:
#             ax.scatter(future_times[diet_future_idx], future_glucose.iloc[diet_future_idx],
#                       c='green', s=80, marker='^', alpha=0.9, zorder=5)
#
#         if len(med_future_idx) > 0:
#             ax.scatter(future_times[med_future_idx], future_glucose.iloc[med_future_idx],
#                       c='red', s=80, marker='o', alpha=0.9, zorder=5)
#
#         if len(ex_future_idx) > 0:
#             ax.scatter(future_times[ex_future_idx], future_glucose.iloc[ex_future_idx],
#                       c='orange', s=80, marker='s', alpha=0.9, zorder=5)
#
#     # Highlight the selected event for withEvent tasks
#     if show_event_marker and hasattr(task, 'selected_event'):
#         event_info = task.selected_event
#         event_time = boundary_time  # The event is at the boundary
#
#         # Get glucose value at boundary
#         boundary_glucose = future_glucose.iloc[0]
#
#         # Event marker
#         event_color = {'Diet5Min': 'green', 'Med5Min': 'red', 'Exercise5Min': 'orange'}
#         event_marker = {'Diet5Min': '^', 'Med5Min': 'o', 'Exercise5Min': 's'}
#
#         color = event_color.get(event_info['type'], 'purple')
#         marker = event_marker.get(event_info['type'], '*')
#         event_name = event_info['type'].replace('5Min', '')
#
#         ax.scatter([event_time], [boundary_glucose], c=color, s=200,
#                   marker=marker, alpha=1.0, zorder=6,
#                   edgecolors='black', linewidth=2,
#                   label=f'Selected {event_name} Event')
#
#     # Add glucose range thresholds
#     ax.axhline(y=70, color='gray', linestyle='--', alpha=0.3, label='Low glucose (70 mg/dL)')
#     ax.axhline(y=180, color='gray', linestyle='--', alpha=0.3, label='High glucose (180 mg/dL)')
#
#     # Formatting
#     ax.set_xlabel('Time', fontsize=12)
#     ax.set_ylabel('Glucose (mg/dL)', fontsize=12)
#     ax.set_title(f'CGM Task Instance - {task.__class__.__name__}', fontsize=14, fontweight='bold')
#
#     # Rotate x-axis labels
#     plt.xticks(rotation=45, ha='right')
#
#     # Add grid
#     ax.grid(True, alpha=0.2)
#
#     # Set y-axis limits
#     ax.set_ylim(0, 450)
#
#     # Legend
#     ax.legend(loc='upper right', fontsize=10)
#
#     # Add stats and info box
#     stats_text = f'Past: {len(past_glucose)} points\n'
#     stats_text += f'Future: {len(future_glucose)} points\n'
#     if hasattr(task, 'selected_event'):
#         stats_text += f'Event: {task.selected_event["type"].replace("5Min", "")}'
#
#     ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
#            fontsize=10, verticalalignment='top',
#            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
#
#     plt.tight_layout()
#     plt.show()
#
#     # Print context information
#     print("\n" + "="*60)
#     print("BACKGROUND:")
#     print(task.background)
#     print("\nSCENARIO:")
#     print(task.scenario)
#     if hasattr(task, 'selected_event'):
#         print(f"\nSELECTED EVENT:")
#         print(f"- Type: {task.selected_event['type']}")
#         print(f"- Index: {task.selected_event['index']}")
#         print(f"- Patient: {task.selected_event['patient_id']}")
#     print("="*60)

def visualize_task_instance(
    task,
    show_event_marker: bool = True,
    figsize: tuple = (15, 6),
    show_interventions: bool = True
):
    """
    Visualize a GlucoseCGMTask or GlucoseCGMTask_withEvent instance.

    Parameters:
    -----------
    task : GlucoseCGMTask or GlucoseCGMTask_withEvent
        The task instance after calling random_instance()
    show_event_marker : bool
        Whether to show the selected event marker (for withEvent tasks)
    figsize : tuple
        Figure size (will be adjusted to 12:5 ratio)
    show_interventions : bool
        Whether to show intervention markers
    """
    # Override figsize to maintain 12:5 ratio
    figsize = (12, 5)

    # Get glucose data
    past_glucose = task.past_time['glucose_mg_dl']
    future_glucose = task.future_time['glucose_mg_dl']

    # Create combined time series for visualization
    combined_time = pd.concat([past_glucose, future_glucose])

    fig, ax = plt.subplots(figsize=figsize)

    # Plot historical data
    ax.plot(past_glucose.index, past_glucose.values, 'b-',
           linewidth=3, alpha=0.8, label='Historical glucose')

    # Plot future data (ground truth)
    ax.plot(future_glucose.index, future_glucose.values, color='orange', linestyle='--',
           linewidth=3, alpha=0.8, label='Future glucose (ground truth)')

    # Add vertical line at prediction boundary
    boundary_time = future_glucose.index[0]
    ax.axvline(x=boundary_time, color='red', linestyle='-', alpha=0.8,
              linewidth=2.5, label='Prediction boundary')

    # For GlucoseCGMTask_withEvent_withLag, add event line and lag visualization
    if hasattr(task, 'selected_event'):
        event_info = task.selected_event
        event_idx = event_info.get('prediction_time_idx', event_info.get('index'))  # Support both old and new
        lag = event_info.get('lag', 0)

        # Calculate the actual event time in the combined series
        # The event_idx is from the original full series (prediction_time_idx)
        # We need to map it to our current window

        # Get the observation point (cut index)
        obs_point = event_info.get('obs_point', event_idx)

        # The past window is [obs_point - C, obs_point)
        # The future window is [obs_point, obs_point + H)
        C = len(past_glucose)

        # Event position relative to observation point
        event_pos_relative = event_idx - (obs_point - C)

        # If event is within our visualization window
        all_times = pd.concat([past_glucose.index.to_series(), future_glucose.index.to_series()])
        all_glucose = pd.concat([past_glucose, future_glucose])

        if 0 <= event_pos_relative < len(all_times):
            event_time = all_times.iloc[event_pos_relative]
            event_glucose = all_glucose.iloc[event_pos_relative]

            # Add vertical line at event time
            # ax.axvline(x=event_time, color='purple', linestyle='--', alpha=0.7,
            #           linewidth=2, label=f'Event ({event_info["type"].replace("5Min", "")})')

            # Add event marker on the glucose curve
            event_color = {'Diet5Min': 'green', 'Med5Min': 'red', 'Exercise5Min': 'orange'}
            event_marker = {'Diet5Min': '^', 'Med5Min': 'o', 'Exercise5Min': 's'}

            color = event_color.get(event_info['type'], 'purple')
            marker = event_marker.get(event_info['type'], '*')

            ax.scatter([event_time], [event_glucose], c=color, s=300,
                      marker=marker, alpha=1.0, zorder=7,
                      edgecolors='black', linewidth=2.5)

            # If there's a lag, show it visually
            if lag != 0:
                # Add arrow showing lag relationship
                if lag > 0:
                    # Event is in the past relative to prediction boundary
                    ax.annotate('', xy=(boundary_time, 50), xytext=(event_time, 50),
                               arrowprops=dict(arrowstyle='<->', color='black', lw=1.5, alpha=0.6))
                    ax.text((pd.Timestamp(event_time) + (pd.Timestamp(boundary_time) - pd.Timestamp(event_time))/2),
                           60, f'Lag: {abs(lag)*5} min', ha='center', fontsize=9)
                elif lag < 0:
                    # Event is in the future relative to prediction boundary
                    ax.annotate('', xy=(event_time, 50), xytext=(boundary_time, 50),
                               arrowprops=dict(arrowstyle='<->', color='black', lw=1.5, alpha=0.6))
                    ax.text((pd.Timestamp(boundary_time) + (pd.Timestamp(event_time) - pd.Timestamp(boundary_time))/2),
                           60, f'Lag: {abs(lag)*5} min', ha='center', fontsize=9)

    # Show all interventions if available
    if show_interventions and hasattr(task, 'c_cov'):
        # Get intervention data
        past_cov = task.c_cov['past']
        future_cov = task.c_cov['future']

        # Last 3 columns are diet, med, exercise
        past_diet = past_cov[:, -3]
        past_med = past_cov[:, -2]
        past_ex = past_cov[:, -1]

        future_diet = future_cov[:, -3]
        future_med = future_cov[:, -2]
        future_ex = future_cov[:, -1]

        # Plot intervention markers
        past_times = past_glucose.index
        future_times = future_glucose.index

        # Past interventions
        diet_past_idx = np.where(past_diet > 0)[0]
        med_past_idx = np.where(past_med > 0)[0]
        ex_past_idx = np.where(past_ex > 0)[0]

        if len(diet_past_idx) > 0:
            ax.scatter(past_times[diet_past_idx], past_glucose.iloc[diet_past_idx],
                      c='green', s=150, marker='^', alpha=0.7, zorder=5, label='Diet')

        if len(med_past_idx) > 0:
            ax.scatter(past_times[med_past_idx], past_glucose.iloc[med_past_idx],
                      c='red', s=150, marker='o', alpha=0.7, zorder=5, label='Medication')

        if len(ex_past_idx) > 0:
            ax.scatter(past_times[ex_past_idx], past_glucose.iloc[ex_past_idx],
                      c='orange', s=150, marker='s', alpha=0.7, zorder=5, label='Exercise')

        # Future interventions
        diet_future_idx = np.where(future_diet > 0)[0]
        med_future_idx = np.where(future_med > 0)[0]
        ex_future_idx = np.where(future_ex > 0)[0]

        if len(diet_future_idx) > 0:
            ax.scatter(future_times[diet_future_idx], future_glucose.iloc[diet_future_idx],
                      c='green', s=150, marker='^', alpha=0.7, zorder=5)

        if len(med_future_idx) > 0:
            ax.scatter(future_times[med_future_idx], future_glucose.iloc[med_future_idx],
                      c='red', s=150, marker='o', alpha=0.7, zorder=5)

        if len(ex_future_idx) > 0:
            ax.scatter(future_times[ex_future_idx], future_glucose.iloc[ex_future_idx],
                      c='orange', s=150, marker='s', alpha=0.7, zorder=5)

    # Add glucose range thresholds
    ax.axhline(y=70, color='gray', linestyle='--', alpha=0.3, label='Low glucose (70 mg/dL)')
    ax.axhline(y=180, color='gray', linestyle='--', alpha=0.3, label='High glucose (180 mg/dL)')

    # Formatting
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Glucose (mg/dL)', fontsize=12)
    ax.set_title(f'CGM Task Instance - {task.__class__.__name__}', fontsize=14, fontweight='bold')

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')

    # Add grid
    ax.grid(True, alpha=0.2)

    # Legend
    ax.legend(bbox_to_anchor=(1.01, 1.05), loc='upper left', fontsize=9)

    # Add stats and info box
    stats_text = f'Past: {len(past_glucose)} points\n'
    stats_text += f'Future: {len(future_glucose)} points\n'
    if hasattr(task, 'selected_event'):
        event_info = task.selected_event
        stats_text += f'Event: {event_info["type"].replace("5Min", "")}\n'
        stats_text += f'Lag: {event_info.get("lag", 0)*5} min'

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Set y-axis limits - MUST be done after all plotting and before tight_layout
    ax.set_ylim(0, 450)

    plt.tight_layout()

    # Ensure y-axis limits are still 0-450 after tight_layout
    ax.set_ylim(0, 450)

    #plt.show()

    # Print context information
    print("\n" + "="*60)
    print("BACKGROUND:")
    print(task.background)
    print("\nSCENARIO:")
    print(task.scenario)
    if hasattr(task, 'selected_event'):
        print(f"\nSELECTED EVENT:")
        print(f"- Type: {task.selected_event['type']}")
        event_idx = task.selected_event.get('prediction_time_idx', task.selected_event.get('index', 'N/A'))
        print(f"- Prediction Time Index: {event_idx}")
        print(f"- Lag: {task.selected_event.get('lag', 0)*5} minutes")
        print(f"- Patient: {task.selected_event['patient_id']}")
    print("="*60)


def plot_cik_style_forecast(
    samples: np.ndarray,
    past_series: pd.Series | None = None,       # e.g., task.past_time["glucose_mg_dl"]
    future_truth: pd.Series | None = None,      # e.g., task.future_time["glucose_mg_dl"]
    title: str | None = None,
    max_sample_traces: int = 10,                # draw at most this many sample paths
):
    """
    Plot forecast samples like the CiK paper: sample paths, mean line, and 50%/90% bands.

    Parameters
    ----------
    samples : array, shape (n_samples, H)
        Forecast sample paths for the next H steps.
    past_series : optional pandas Series (length C)
        Historical target values with a DatetimeIndex (preferred) or RangeIndex.
    future_truth : optional pandas Series (length H)
        Ground-truth future values (if available) with matching index to samples’ horizon.
    title : str
        Plot title.
    max_sample_traces : int
        Limit how many individual sample paths to draw (for readability).
    """
    assert samples.ndim == 2, "samples must be (n_samples, H)"
    n_samples, H = samples.shape

    # Compute summary stats
    mean = samples.mean(axis=0)
    q05, q25 = np.percentile(samples, [5, 25], axis=0)
    q75, q95 = np.percentile(samples, [75, 95], axis=0)

    # Build x-axes
    if future_truth is not None and isinstance(future_truth.index, pd.DatetimeIndex):
        x_future = future_truth.index
    else:
        x_future = np.arange(H)

    # Prepare figure
    plt.figure(figsize=(10, 5))

    # 1) Past history
    if past_series is not None:
        if isinstance(past_series, pd.Series):
            x_past = past_series.index if isinstance(past_series.index, (pd.DatetimeIndex, pd.RangeIndex)) else np.arange(len(past_series))
            plt.plot(x_past, past_series.values, color='blue', linewidth=3, label="past")
        else:
            # If a DataFrame was passed accidentally, use its last column
            vals = np.asarray(past_series)
            plt.plot(np.arange(vals.shape[0]), vals.squeeze(), color='blue', linewidth=3, label="past")

    # 2) Sample paths (thin)
    n_traces = min(max_sample_traces, n_samples)
    # pick evenly spaced examples across the sample set
    if n_traces > 0:
        idx = np.linspace(0, n_samples - 1, n_traces).astype(int)
        for i in idx:
            plt.plot(x_future, samples[i], linewidth=0.7, alpha=0.6)

    # 3) Uncertainty bands
    plt.fill_between(x_future, q05, q95, alpha=0.25, label="90% band")
    plt.fill_between(x_future, q25, q75, alpha=0.4, label="50% band")

    # 4) Predictive mean
    plt.plot(x_future, mean, color='orange', linewidth=3, label="predictive mean")

    # 5) Future ground truth
    if future_truth is not None:
        plt.plot(x_future, future_truth.values, color='gray', linewidth=2, linestyle="--", label="future truth")

    # Cosmetics
    if title:
        plt.title(title)
    plt.xlabel("time")
    plt.ylabel("value")
    plt.legend(bbox_to_anchor=(1.01, 1.05), loc='upper left')
    plt.tight_layout()
    # plt.show()



