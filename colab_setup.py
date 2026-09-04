"""
Google Colab Setup & One-Click Runner (MEGH-DRISHTI)
---------------------------------------------------
Automates running the AI Nowcasting Engine and Interactive Dashboard inside Google Colab.
"""

import os
import sys
import subprocess
import time
import threading

def start_background_server():
    """Starts the early warning system backend in a daemon thread."""
    from run_early_warning_system import main as run_server
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    print("[✓] AI Nowcasting Server launched in background!")
    time.sleep(2)

def launch_colab():
    print("=" * 70)
    print("⛈️  MEGH-DRISHTI: Google Colab Runner")
    print("=" * 70)
    
    # 1. Start backend server
    start_background_server()
    
    # 2. Check if running inside Google Colab environment
    try:
        from google.colab.output import serve_kernel_port_as_window, serve_kernel_port_as_iframe
        print("\n[+] Detected Google Colab Environment!")
        print("[+] Click the link below to open the Live Command-Center Dashboard in a new tab:")
        serve_kernel_port_as_window(8000)
        
        print("\n[+] Or previewing inline below:")
        serve_kernel_port_as_iframe(8000, width="100%", height="750")
    except ImportError:
        print("\n[!] Not in Google Colab. Running locally on http://localhost:8000")
        
    # Keep server process running indefinitely
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[!] Stopping server.")

if __name__ == "__main__":
    launch_colab()
