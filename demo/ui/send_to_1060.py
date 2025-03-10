import csv
import socket
import time
import tkinter as tk
from tkinter import filedialog, messagebox

def send_data():
    # Get file path from the entry widget
    file_path = file_entry.get()
    
    if not file_path:
        messagebox.showerror("Error", "Please select a CSV file")
        return
    try:
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(('192.168.0.200', 7))
                for row in reader:
                    message = ','.join(row)
                    s.sendall(message.encode('utf-8')+b"\n")
                    print(f"Sent: {message}")
                    time.sleep(1.6)
    except Exception as e:
        messagebox.showerror("Error", str(e))

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    file_entry.delete(0, tk.END)
    file_entry.insert(0, file_path)

# Create the main window
root = tk.Tk()
root.title("CSV Sender")

# Create and place the widgets
file_label = tk.Label(root, text="CSV File:")
file_label.grid(row=0, column=0, padx=10, pady=10)

file_entry = tk.Entry(root, width=50)
file_entry.grid(row=0, column=1, padx=10, pady=10)

browse_button = tk.Button(root, text="Browse", command=browse_file)
browse_button.grid(row=0, column=2, padx=10, pady=10)

send_button = tk.Button(root, text="Send", command=send_data)
send_button.grid(row=1, column=1, padx=10, pady=10)

# Run the main event loop
root.mainloop()
