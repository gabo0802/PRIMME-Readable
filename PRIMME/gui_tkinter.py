#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 01:12:31 2026

@author: gabriel.castejon
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import subprocess
import sys
import os
from pathlib import Path
import threading
import time
import matplotlib.pyplot as plt
import glob
from PIL import Image, ImageTk

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))   
os.chdir(__location__) # ensure that the working directory is where this scipt is located.

fp = './data/'
if not os.path.exists(fp): os.makedirs(fp)

fp = './plots/'
if not os.path.exists(fp): os.makedirs(fp)

# Function to run the script with the selected parameters
def run_primme_simulation(parameters, console_output, stop_event):
    # Build command with parameters
    cmd = [sys.executable, "run_script.py"]
    for key, value in parameters.items():
        if value is not None and value != "":
            if isinstance(value, bool) and value:
                cmd.append(f"--{key}")
            elif not isinstance(value, bool):
                cmd.append(f"--{key}={value}")
    
    # Create process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Create a separate thread to monitor the stop_event
    def monitor_stop_event():
        while process.poll() is None:  # While process is running
            if stop_event.is_set():
                # Try a more graceful shutdown approach
                try:
                    import signal
                    # Send SIGINT instead of SIGTERM
                    process.send_signal(signal.SIGINT)
                    # Give it some time to clean up
                    time.sleep(0.5)
                    # Only force terminate if it's still running
                    if process.poll() is None:
                        process.terminate()
                except Exception as e:
                    push_to_console(f"Error during termination: {str(e)}")
                push_to_console("Process terminated by user.")
                break
            time.sleep(0.1)  # Check every 100ms
    
    monitor_thread = threading.Thread(target=monitor_stop_event)
    monitor_thread.daemon = True
    monitor_thread.start()

    # Read and display output in real-time
    try:
        for line in iter(process.stdout.readline, ''):
            if stop_event.is_set():
                break
            # Use after to update UI from a different thread
            root.after(0, lambda l=line: push_to_console(l))
    except Exception as e:
        root.after(0, lambda: push_to_console(f"Error reading output: {str(e)}"))
    finally:
        # Ensure resources are properly closed
        try:
            process.stdout.close()
        except:
            pass
    
    return_code = process.wait()
    
    if return_code == 0 and not stop_event.is_set():
        root.after(0, lambda: push_to_console("Process completed successfully!"))
    elif return_code != 0 and not stop_event.is_set():
        root.after(0, lambda: push_to_console(f"Process failed with return code {return_code}"))

# Display each plot
def format_plot_title(filename):
    stem = Path(filename).stem
    
    # Remove everything before and including the closing parenthesis if it exists
    if ')' in stem:
        stem = stem.split(')')[-1]
    
    # Replace underscores with spaces and capitalize
    return stem.replace('_', ' ').title()

def push_to_console(text):
    console_output.configure(state="normal")
    console_output.insert(tk.END, text + "\n")
    console_output.see(tk.END)
    console_output.configure(state="disabled")
    
def clear_console():
    console_output.configure(state="normal")
    console_output.delete(1.0, tk.END)
    console_output.configure(state="disabled")

# Main application
def create_app():
    global root, console_output, parameters
    
    # Initialize parameters with defaults
    parameters = {
        "trainset": "./data/trainset_spparks_sz(257x257)_ng(256-256)_nsets(200)_future(4)_max(100)_kt(0.66)_cut(0).h5",
        "modelname": None,
        "dims": 2,
        "if_plot": False,
        "num_eps": 1000,
        "obs_dim": 17,
        "act_dim": 17,
        "lr": 5e-5,
        "reg": 1,
        "nsteps": 1000,
        "n_samples": 200,
        "mode": "Single_Step",
        "grain_shape": "grain",
        "grain_size": 512,
        "voroni_loaded": False,
        "ic": "./data/ic.npy",
        "ea": "./data/ea.npy",
        "ma": "./data/ma.npy",
        "ic_shape": "grain(512_512_512)",
        "size": 93,
        "dimension": 2,
        "ngrain": 2**10,
        "primme": None,
        "pad_mode": "circular",
        "if_output_plot": False
    }
    
    data_files = []
    for ext in ['*.h5']:
        data_files.extend(glob.glob(f"./data/{ext}"))
    spparks_trainsets = [f for f in data_files if 'spparks' in f]
    models_trained = [f for f in data_files if 'model' in f]
    primme_simulations = [f for f in data_files if 'primme' in f]
    
    # Create root window
    root = tk.Tk()
    root.title("PRIMME Simulation")
    root.geometry("900x700")
    
    # Create notebook (tabs)
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=10, pady=5)
    
    # Create tab frames
    model_tab = ttk.Frame(notebook)
    grain_tab = ttk.Frame(notebook)
    run_tab = ttk.Frame(notebook)
    results_tab = ttk.Frame(notebook)
    
    # Add tabs to notebook
    notebook.add(model_tab, text="Training Parameters")
    notebook.add(grain_tab, text="Testing Parameters")
    notebook.add(run_tab, text="Run Simulation")
    notebook.add(results_tab, text="Results")
    
    # Training Parameters Tab
    model_frame = ttk.LabelFrame(model_tab, text="Training Parameters")
    model_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Training set selection
    ttk.Label(model_frame, text="Training Set").grid(row=0, column=0, sticky="w", padx=5, pady=5)
    trainset_var = tk.StringVar(value=parameters["trainset"])
    trainset_combo = ttk.Combobox(model_frame, textvariable=trainset_var, values=spparks_trainsets, width=50)
    trainset_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
    
    def on_trainset_change(event):
        parameters.update({"trainset": trainset_var.get()})
    
    trainset_combo.bind("<<ComboboxSelected>>", on_trainset_change)
    
    # Model name selection
    ttk.Label(model_frame, text="Model Name").grid(row=1, column=0, sticky="w", padx=5, pady=5)
    model_options = ['Train New Model'] + models_trained
    modelname_var = tk.StringVar(value='Train New Model')
    modelname_combo = ttk.Combobox(model_frame, textvariable=modelname_var, values=model_options, width=50)
    modelname_combo.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
    
    # Container for model parameters (shown only when "Train New Model" is selected)
    model_params_frame = ttk.Frame(model_frame)
    model_params_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
    
    # Observation and Action dimensions
    params_row1 = ttk.Frame(model_params_frame)
    params_row1.pack(fill="x", pady=5)
    
    ttk.Label(params_row1, text="Observation Dimension").pack(side="left", padx=5)
    obs_dim_var = tk.IntVar(value=parameters["obs_dim"])
    obs_dim_combo = ttk.Combobox(params_row1, textvariable=obs_dim_var, values=[7,9,11,13,15,17,19,21], width=5)
    obs_dim_combo.pack(side="left", padx=5)
    
    ttk.Label(params_row1, text="Action Dimension").pack(side="left", padx=20)
    act_dim_var = tk.IntVar(value=parameters["act_dim"])
    act_dim_combo = ttk.Combobox(params_row1, textvariable=act_dim_var, values=[7,9,11,13,15,17,19,21], width=5)
    act_dim_combo.pack(side="left", padx=5)
    
    # Number of epochs slider
    params_row2 = ttk.Frame(model_params_frame)
    params_row2.pack(fill="x", pady=10)
    
    num_eps_label = ttk.Label(params_row2, text=f"Number of Training Epochs: {parameters['num_eps']}")
    num_eps_label.pack(anchor="w", padx=5)
    
    num_eps_var = tk.IntVar(value=parameters["num_eps"])
    num_eps_slider = ttk.Scale(params_row2, from_=5, to=2000, variable=num_eps_var, orient="horizontal")
    num_eps_slider.pack(fill="x", padx=5)
    
    def update_num_eps(event):
        value = int(num_eps_slider.get())
        parameters.update({"num_eps": value})
        num_eps_label.config(text=f"Number of Training Epochs: {value}")
    
    num_eps_slider.bind("<ButtonRelease-1>", update_num_eps)
    
    # Padding mode
    params_row3 = ttk.Frame(model_params_frame)
    params_row3.pack(fill="x", pady=5)
    
    ttk.Label(params_row3, text="Padding Mode").pack(side="left", padx=5)
    pad_mode_var = tk.StringVar(value=parameters["pad_mode"])
    pad_mode_combo = ttk.Combobox(params_row3, textvariable=pad_mode_var, values=["circular", "reflect"], width=10)
    pad_mode_combo.pack(side="left", padx=5)
    
    def on_pad_mode_change(event):
        parameters.update({"pad_mode": pad_mode_var.get()})
    
    pad_mode_combo.bind("<<ComboboxSelected>>", on_pad_mode_change)
    
    # Output plots checkbox
    if_plot_var = tk.BooleanVar(value=parameters["if_plot"])
    if_plot_check = ttk.Checkbutton(model_params_frame, text="Output Plots During Training", 
                                   variable=if_plot_var, 
                                   command=lambda: parameters.update({"if_plot": if_plot_var.get()}))
    if_plot_check.pack(anchor="w", padx=5, pady=5)
    
    # Function to show/hide model parameters based on selection
    def on_modelname_change(event):
        selected = modelname_var.get()
        if selected == "Train New Model":
            parameters.update({"modelname": None})
            model_params_frame.pack(fill="both", expand=True, padx=5, pady=5)
        else:
            parameters.update({"modelname": selected})
            model_params_frame.pack_forget()
    
    modelname_combo.bind("<<ComboboxSelected>>", on_modelname_change)
    
    # Testing Parameters Tab
    grain_frame = ttk.LabelFrame(grain_tab, text="Testing Parameters")
    grain_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    # PRIMME simulation selection
    ttk.Label(grain_frame, text="PRIMME Simulation").grid(row=0, column=0, sticky="w", padx=5, pady=5)
    primme_options = ["Run New Model"] + primme_simulations
    primme_var = tk.StringVar(value="Run New Model")
    primme_combo = ttk.Combobox(grain_frame, textvariable=primme_var, values=primme_options, width=50)
    primme_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
    
    # Container for grain parameters
    grain_params_frame = ttk.Frame(grain_frame)
    grain_params_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
    
    # Voroni loaded checkbox
    voroni_var = tk.BooleanVar(value=parameters["voroni_loaded"])
    voroni_check = ttk.Checkbutton(grain_params_frame, text="Voroni Loaded", 
                                  variable=voroni_var, 
                                  command=lambda: parameters.update({"voroni_loaded": voroni_var.get()}))
    voroni_check.pack(anchor="w", padx=5, pady=5)
    
    # Non-loaded inputs frame
    non_loaded_frame = ttk.Frame(grain_params_frame)
    non_loaded_frame.pack(fill="x", pady=5)
    
    ttk.Label(non_loaded_frame, text="Grain Size").pack(side="left", padx=5)
    grain_size_var = tk.IntVar(value=parameters["grain_size"])
    grain_size_combo = ttk.Combobox(non_loaded_frame, textvariable=grain_size_var, 
                                   values=[257, 512, 1024, 2048, 2400], width=10)
    grain_size_combo.pack(side="left", padx=5)
    
    ttk.Label(non_loaded_frame, text="Grain Shape").pack(side="left", padx=20)
    grain_shape_var = tk.StringVar(value=parameters["grain_shape"])
    grain_shape_combo = ttk.Combobox(non_loaded_frame, textvariable=grain_shape_var, 
                                    values=["grain", "circular", "hex", "square"], width=10)
    grain_shape_combo.pack(side="left", padx=5)
    
    # Loaded inputs frame
    loaded_frame = ttk.Frame(grain_params_frame)
    
    ttk.Label(loaded_frame, text="Initial Condition Path").grid(row=0, column=0, sticky="w", padx=5, pady=5)
    ic_var = tk.StringVar(value=parameters["ic"])
    ic_entry = ttk.Entry(loaded_frame, textvariable=ic_var, width=50)
    ic_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
    
    ttk.Label(loaded_frame, text="Euler Angles Path").grid(row=1, column=0, sticky="w", padx=5, pady=5)
    ea_var = tk.StringVar(value=parameters["ea"])
    ea_entry = ttk.Entry(loaded_frame, textvariable=ea_var, width=50)
    ea_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
    
    ttk.Label(loaded_frame, text="Miso Angles Path").grid(row=2, column=0, sticky="w", padx=5, pady=5)
    ma_var = tk.StringVar(value=parameters["ma"])
    ma_entry = ttk.Entry(loaded_frame, textvariable=ma_var, width=50)
    ma_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
    
    # Number of grains slider
    ngrain_frame = ttk.Frame(grain_params_frame)
    ngrain_frame.pack(fill="x", pady=10)
    
    ngrain_exponent = parameters["ngrain"].bit_length() - 1
    ngrain_label = ttk.Label(ngrain_frame, text=f"Number of Grains: 2^{ngrain_exponent} or {parameters['ngrain']}")
    ngrain_label.pack(anchor="w", padx=5)
    
    ngrain_var = tk.IntVar(value=ngrain_exponent)
    ngrain_slider = ttk.Scale(ngrain_frame, from_=6, to=18, variable=ngrain_var, orient="horizontal")
    ngrain_slider.pack(fill="x", padx=5)
    
    def update_ngrain(event):
        exponent = int(ngrain_slider.get())
        ngrain = 2 ** exponent
        parameters.update({"ngrain": ngrain})
        ngrain_label.config(text=f"Number of Grains: 2^{exponent} or {ngrain}")
    
    ngrain_slider.bind("<ButtonRelease-1>", update_ngrain)
    
    # Number of steps slider
    nsteps_frame = ttk.Frame(grain_params_frame)
    nsteps_frame.pack(fill="x", pady=10)
    
    nsteps_label = ttk.Label(nsteps_frame, text=f"Number of Steps: {parameters['nsteps']}")
    nsteps_label.pack(anchor="w", padx=5)
    
    nsteps_var = tk.IntVar(value=parameters["nsteps"])
    nsteps_slider = ttk.Scale(nsteps_frame, from_=10, to=1000, variable=nsteps_var, orient="horizontal")
    nsteps_slider.pack(fill="x", padx=5)
    
    def update_nsteps(event):
        value = int(nsteps_slider.get())
        parameters.update({"nsteps": value})
        nsteps_label.config(text=f"Number of Steps: {value}")
    
    nsteps_slider.bind("<ButtonRelease-1>", update_nsteps)
    
    # Function to update grain related UI elements
    def update_grain(event=None):
        shape = grain_shape_var.get()
        if shape == 'grain':
            # Enable the slider and reset to default value
            ngrain_slider.state(['!disabled'])
            ngrain_var.set(10)
            parameters.update({"ngrain": 2**10})
            ngrain_label.config(text="Number of Grains: 2^10 or 1024")
            
            grain_size_combo.state(['!disabled'])
            grain_size_var.set(512)
            parameters.update({"grain_size": 512})
        else:
            if shape == 'hex':
                grain_size_var.set(443)
                grain_size_combo.state(['disabled'])
                parameters.update({"grain_size": 443})
    
    grain_shape_combo.bind("<<ComboboxSelected>>", update_grain)
    
    # Function to show/hide frames based on voroni checkbox
    def toggle_voroni_frames():
        if voroni_var.get():
            loaded_frame.pack(fill="x", pady=5)
            non_loaded_frame.pack_forget()
        else:
            non_loaded_frame.pack(fill="x", pady=5)
            loaded_frame.pack_forget()
    
    voroni_check.config(command=lambda: (parameters.update({"voroni_loaded": voroni_var.get()}), toggle_voroni_frames()))
    
    # Function to show/hide grain parameters based on PRIMME selection
    def on_primme_change(event):
        selected = primme_var.get()
        if selected == "Run New Model":
            parameters.update({"primme": None})
            grain_params_frame.pack(fill="both", expand=True, padx=5, pady=5)
        else:
            parameters.update({"primme": selected})
            grain_params_frame.pack_forget()
    
    primme_combo.bind("<<ComboboxSelected>>", on_primme_change)
    
    # Initialize visibility
    toggle_voroni_frames()
    
    # Run Simulation Tab
    run_frame = ttk.LabelFrame(run_tab, text="Run Simulation")
    run_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Console output
    console_frame = ttk.Frame(run_frame)
    console_frame.pack(fill="both", expand=True, padx=5, pady=5)
    
    console_output = scrolledtext.ScrolledText(console_frame, height=20, bg="black", fg="green", font=("Courier", 10))
    console_output.pack(fill="both", expand=True)
    console_output.configure(state="disabled")
    
    # Buttons frame
    buttons_frame = ttk.Frame(run_frame)
    buttons_frame.pack(fill="x", padx=5, pady=10)
    
    stop_event = threading.Event()
    
    def on_run_click():
        clear_console()
        push_to_console("Starting PRIMME simulation...")
        stop_event.clear()
        
        # Run in a separate thread to keep UI responsive
        thread = threading.Thread(
            target=run_primme_simulation,
            args=(parameters, console_output, stop_event)
        )
        thread.daemon = True
        thread.start()
    
    def on_run_stop():
        push_to_console("Stopping PRIMME simulation...")
        stop_event.set()
    
    run_button = ttk.Button(buttons_frame, text="Run Simulation", command=lambda: (on_run_click(), update_buttons(1)))
    run_button.pack(side="left", padx=5)
    
    stop_button = ttk.Button(buttons_frame, text="Stop Simulation", command=lambda: (on_run_stop(), update_buttons(0)))
    stop_button.pack(side="left", padx=5)
    
    def update_buttons(button=0):
        if button:
            run_button.state(['disabled'])
            stop_button.state(['!disabled'])
        else:
            run_button.state(['!disabled'])
            stop_button.state(['disabled'])
    
    # Initialize button states
    update_buttons()
    
    # Results Tab
    results_frame = ttk.LabelFrame(results_tab, text="Results Viewer")
    results_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Header row with title and refresh button
    header_frame = ttk.Frame(results_frame)
    header_frame.pack(fill="x", padx=5, pady=5)
    
    refresh_button = ttk.Button(header_frame, text="Refresh Results", 
                               command=lambda: (refresh_results(), refresh_videos()))
    refresh_button.pack(side="right", padx=5)
    
    # Create a canvas with scrollbar for plots
    plot_canvas_frame = ttk.Frame(results_frame)
    plot_canvas_frame.pack(fill="both", expand=True, padx=5, pady=5)
    
    plot_canvas = tk.Canvas(plot_canvas_frame)
    plot_scrollbar = ttk.Scrollbar(plot_canvas_frame, orient="vertical", command=plot_canvas.yview)
    plot_canvas.configure(yscrollcommand=plot_scrollbar.set)
    
    plot_scrollbar.pack(side="right", fill="y")
    plot_canvas.pack(side="left", fill="both", expand=True)
    
    plot_container = ttk.Frame(plot_canvas)
    plot_canvas.create_window((0, 0), window=plot_container, anchor="nw")
    
    def on_plot_container_configure(event):
        plot_canvas.configure(scrollregion=plot_canvas.bbox("all"))
    
    plot_container.bind("<Configure>", on_plot_container_configure)
    
    # Video container
    video_container = ttk.Frame(results_frame)
    video_container.pack(fill="both", expand=True, padx=5, pady=5)
    
    def refresh_results():
        # Clear existing plots
        for widget in plot_container.winfo_children():
            widget.destroy()
        
        # Look for plot files
        plot_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            plot_files.extend(glob.glob(f"./plots/{ext}"))
        
        if not plot_files:
            ttk.Label(plot_container, text="No plots found", font=("", 10, "italic")).pack(pady=10)
            return
        
        for plot_file in sorted(plot_files):
            frame = ttk.Frame(plot_container)
            frame.pack(fill="x", pady=10)
            
            ttk.Label(frame, text=format_plot_title(plot_file), font=("", 12, "bold")).pack(anchor="center")
            
            # Load and display the image
            try:
                img = Image.open(plot_file)
                # Resize if too large
                max_width = 800
                if img.width > max_width:
                    ratio = max_width / img.width
                    new_width = max_width
                    new_height = int(img.height * ratio)
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                
                photo = ImageTk.PhotoImage(img)
                img_label = ttk.Label(frame, image=photo)
                img_label.image = photo  # Keep a reference
                img_label.pack(anchor="center", pady=5)
            except Exception as e:
                ttk.Label(frame, text=f"Error loading image: {str(e)}").pack(pady=5)
    
    def refresh_videos():
        # Clear existing videos
        for widget in video_container.winfo_children():
            widget.destroy()
        
        # Look for video files
        video_files = glob.glob('./plots/*.mp4')
        
        if not video_files:
            ttk.Label(video_container, text="No videos found", font=("", 10, "italic")).pack(pady=10)
            return
        
        for video_file in sorted(video_files):
            frame = ttk.Frame(video_container)
            frame.pack(fill="x", pady=10)
            
            ttk.Label(frame, text=format_plot_title(video_file), font=("", 12, "bold")).pack(anchor="center")
            
            # For videos, we'll just show a link to open them externally
            def open_video(file_path=video_file):
                import subprocess
                import platform
                
                if platform.system() == 'Darwin':  # macOS
                    subprocess.call(('open', file_path))
                elif platform.system() == 'Windows':  # Windows
                    os.startfile(file_path)
                else:  # Linux variants
                    subprocess.call(('xdg-open', file_path))
            
            ttk.Button(frame, text="Open Video", command=open_video).pack(anchor="center", pady=5)
    
    # Initial load of results
    refresh_results()
    refresh_videos()
    
    # Main loop
    root.mainloop()

# Run the application
if __name__ == "__main__":
    create_app()

