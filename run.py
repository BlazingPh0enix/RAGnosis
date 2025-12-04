"""
Run script to start both FastAPI and Streamlit servers.

Usage:
    python run.py api        # Start FastAPI only
    python run.py streamlit  # Start Streamlit only  
    python run.py both       # Start both (default)
"""

import subprocess
import sys
import time
import os

# Change to project directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def run_api():
    """Start the FastAPI server."""
    print("🚀 Starting FastAPI server on http://localhost:8000")
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--reload",
        "--host", "0.0.0.0",
        "--port", "8000"
    ])


def run_streamlit():
    """Start the Streamlit server."""
    print("🎨 Starting Streamlit app on http://localhost:8501")
    subprocess.run([
        sys.executable, "-m", "streamlit",
        "run", "app/streamlit_app.py",
        "--server.port", "8501"
    ])


def run_both():
    """Start both servers."""
    import threading
    
    print("=" * 50)
    print("🔍 DocuLens - Multi-Modal RAG System")
    print("=" * 50)
    print()
    print("Starting services...")
    print("  • API:       http://localhost:8000")
    print("  • API Docs:  http://localhost:8000/docs")
    print("  • Streamlit: http://localhost:8501")
    print()
    print("Press Ctrl+C to stop all services")
    print("=" * 50)
    
    # Start API in background
    api_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--host", "0.0.0.0",
        "--port", "8000"
    ])
    
    # Wait a moment for API to start
    time.sleep(2)
    
    # Start Streamlit (blocking)
    try:
        streamlit_process = subprocess.Popen([
            sys.executable, "-m", "streamlit",
            "run", "app/streamlit_app.py",
            "--server.port", "8501"
        ])
        streamlit_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        api_process.terminate()
        print("✅ Services stopped")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "api":
            run_api()
        elif command == "streamlit":
            run_streamlit()
        elif command == "both":
            run_both()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python run.py [api|streamlit|both]")
    else:
        run_both()
