#!/usr/bin/env python3
"""
Run the CV Analysis Web Application

Usage:
    python run_webapp.py [--port PORT] [--host HOST] [--debug]

Examples:
    python run_webapp.py                    # Run on localhost:5000
    python run_webapp.py --port 8080        # Run on localhost:8080
    python run_webapp.py --host 0.0.0.0     # Allow external connections
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []

    try:
        import flask
    except ImportError:
        missing.append('flask')

    try:
        import torch
    except ImportError:
        missing.append('torch')

    try:
        import transformers
    except ImportError:
        missing.append('transformers')

    # Check PDF extraction libraries
    pdf_lib_available = False
    try:
        import fitz  # PyMuPDF
        pdf_lib_available = True
    except ImportError:
        pass

    if not pdf_lib_available:
        try:
            import pdfplumber
            pdf_lib_available = True
        except ImportError:
            pass

    if not pdf_lib_available:
        try:
            import PyPDF2
            pdf_lib_available = True
        except ImportError:
            pass

    if not pdf_lib_available:
        missing.append('PyMuPDF (or pdfplumber or PyPDF2)')

    if missing:
        print("Missing dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall them with:")
        print("  pip install flask torch transformers PyMuPDF")
        return False

    return True


def check_models():
    """Check if trained models exist."""
    model_paths = [
        project_root / 'trained_models' / 'ner' / 'bert' / 'best_model.pt',
        project_root / 'trained_models' / 'cv_jd_matching' / 'bert' / 'best_model.pt',
        project_root / 'trained_models' / 'ats' / 'bert' / 'best_model.pt',
    ]

    missing = []
    for path in model_paths:
        if not path.exists():
            missing.append(str(path))

    if missing:
        print("Missing model files:")
        for path in missing:
            print(f"  - {path}")
        print("\nPlease train the models first or check the paths.")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description='Run CV Analysis Web Application')
    parser.add_argument('--port', type=int, default=5000, help='Port to run on (default: 5000)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind to (default: 127.0.0.1)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--skip-checks', action='store_true', help='Skip dependency and model checks')
    args = parser.parse_args()

    if not args.skip_checks:
        print("Checking dependencies...")
        if not check_dependencies():
            sys.exit(1)

        print("Checking models...")
        if not check_models():
            sys.exit(1)

    print("\n" + "=" * 60)
    print("CV Analysis Web Application")
    print("=" * 60)

    # Import and create app
    from test_web.app import create_app
    app = create_app()

    print(f"\nStarting server on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")

    # Run the app
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )


if __name__ == '__main__':
    main()
