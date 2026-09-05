#!/usr/bin/env python3
"""
Megh-Drishti: Single-command launcher
--------------------------------------
Usage:
    python run_server.py              # starts on port 8000, opens browser
    python run_server.py --port 8080  # custom port
    python run_server.py --no-browser # headless mode
"""

import os
import sys
import argparse
import threading
import webbrowser
import time

# Ensure project root is on path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

def check_deps():
    missing = []
    for pkg in ["fastapi", "uvicorn", "numpy"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\n[!] Missing packages: {missing}")
        venv = os.path.join(ROOT, ".venv")
        if os.path.exists(venv):
            print(f"    Activate your venv:  source {venv}/bin/activate")
        else:
            print(f"    Install:  pip install fastapi uvicorn numpy torch")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Megh-Drishti AI Nowcasting Server")
    parser.add_argument("--port",       type=int,  default=8000, help="Port (default: 8000)")
    parser.add_argument("--host",       type=str,  default="0.0.0.0", help="Host (default: 0.0.0.0)")
    parser.add_argument("--no-browser", action="store_true",    help="Don't auto-open browser")
    args = parser.parse_args()

    check_deps()

    print("\n" + "="*60)
    print("  🌩️   MEGH-DRISHTI  |  AI Hyper-Local Early Warning System")
    print("  🇮🇳   Smart India Hackathon 2026  |  SIH-2026")
    print("="*60)
    print(f"\n  ► Dashboard : http://localhost:{args.port}/dashboard/")
    print(f"  ► API Docs  : http://localhost:{args.port}/docs")
    print(f"  ► Health    : http://localhost:{args.port}/api/health")
    print("\n  Press Ctrl+C to stop\n")

    if not args.no_browser:
        url = f"http://localhost:{args.port}/dashboard/"
        threading.Timer(2.5, lambda: webbrowser.open(url)).start()

    try:
        import uvicorn
        from api.server import app
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    except KeyboardInterrupt:
        print("\n\n[✓] Megh-Drishti server stopped.")
    except Exception as exc:
        print(f"\n[✗] Server error: {exc}")
        raise


if __name__ == "__main__":
    main()
