
import tkinter as tk
from tkinter import messagebox
from predict_weather_from_input import process_inputs

def submit():
    try:
        date = date_entry.get()
        precipitation = float(precipitation_entry.get())
        high_temp = float(high_temp_entry.get())
        low_temp = float(low_temp_entry.get())
        wind = float(wind_entry.get())

        # Call the process_inputs function
        result = process_inputs([date, precipitation, high_temp, low_temp, wind])
        messagebox.showinfo("Prediction Result", f"The predicted weather is: {result}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def center_window(root, width, height):
    # Get screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Calculate position to center the window
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)

    # Set the window size and position
    root.geometry(f'{width}x{height}+{x}+{y}')

# Create the Tkinter window
root = tk.Tk()
root.title("Weather Prediction")

# Set the desired window size and center it
window_width = 400
window_height = 300
center_window(root, window_width, window_height)

# Create the input fields
tk.Label(root, text="Date (YYYY-MM-DD):").grid(row=0, column=0, padx=10, pady=10)
date_entry = tk.Entry(root)
date_entry.grid(row=0, column=1, padx=10, pady=10)

tk.Label(root, text="Precipitation:").grid(row=1, column=0, padx=10, pady=10)
precipitation_entry = tk.Entry(root)
precipitation_entry.grid(row=1, column=1, padx=10, pady=10)

tk.Label(root, text="High Temperature:").grid(row=2, column=0, padx=10, pady=10)
high_temp_entry = tk.Entry(root)
high_temp_entry.grid(row=2, column=1, padx=10, pady=10)

tk.Label(root, text="Low Temperature:").grid(row=3, column=0, padx=10, pady=10)
low_temp_entry = tk.Entry(root)
low_temp_entry.grid(row=3, column=1, padx=10, pady=10)

tk.Label(root, text="Wind:").grid(row=4, column=0, padx=10, pady=10)
wind_entry = tk.Entry(root)
wind_entry.grid(row=4, column=1, padx=10, pady=10)

# Create the submit button
submit_button = tk.Button(root, text="Predict Weather", command=submit)
submit_button.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

# Run the Tkinter main loop
root.mainloop()
#hello
