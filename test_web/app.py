"""
Flask Web Application for CV Analysis
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from pdf_extractor import extract_text_from_pdf
from test_web.inference import get_analyzer


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Configuration
    app.config['SECRET_KEY'] = 'dev-secret-key-change-in-production'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'

    # Ensure upload folder exists
    app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'pdf'}

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    @app.route('/')
    def index():
        """Main page with the analysis form."""
        return render_template('index.html')

    @app.route('/analyze', methods=['POST'])
    def analyze():
        """
        Analyze CV against job description.

        Expects:
        - cv_file: PDF file (optional if cv_text is provided)
        - cv_text: Raw CV text (optional if cv_file is provided)
        - job_description: Job description text
        """
        try:
            # Get job description
            job_description = request.form.get('job_description', '').strip()
            if not job_description:
                return jsonify({'error': 'Job description is required'}), 400

            # Get CV text - either from file or text input
            cv_text = ''

            # Check for PDF file upload
            if 'cv_file' in request.files:
                file = request.files['cv_file']
                if file and file.filename and allowed_file(file.filename):
                    try:
                        # Extract text from PDF
                        pdf_bytes = file.read()
                        cv_text = extract_text_from_pdf(pdf_bytes)
                    except Exception as e:
                        return jsonify({'error': f'Failed to extract text from PDF: {str(e)}'}), 400

            # Fall back to text input if no file or file is empty
            if not cv_text:
                cv_text = request.form.get('cv_text', '').strip()

            if not cv_text:
                return jsonify({'error': 'Please provide a CV (either upload PDF or paste text)'}), 400

            # Get analyzer and run analysis
            analyzer = get_analyzer()
            results = analyzer.analyze(cv_text, job_description)

            # Add the extracted CV text to results for display
            results['cv_text'] = cv_text[:2000] + '...' if len(cv_text) > 2000 else cv_text

            return jsonify(results)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

    @app.route('/extract-pdf', methods=['POST'])
    def extract_pdf():
        """
        Extract text from uploaded PDF.
        Used to preview extracted text before analysis.
        """
        try:
            if 'cv_file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400

            file = request.files['cv_file']
            if not file or not file.filename:
                return jsonify({'error': 'No file selected'}), 400

            if not allowed_file(file.filename):
                return jsonify({'error': 'Only PDF files are allowed'}), 400

            # Extract text
            pdf_bytes = file.read()
            text = extract_text_from_pdf(pdf_bytes)

            return jsonify({'text': text})

        except Exception as e:
            return jsonify({'error': f'Failed to extract text: {str(e)}'}), 500

    @app.route('/health')
    def health():
        """Health check endpoint."""
        return jsonify({'status': 'healthy'})

    return app


# For running directly
if __name__ == '__main__':
    app = create_app()
    print("\n" + "=" * 60)
    print("CV Analysis Web Application")
    print("=" * 60)
    print("\nStarting server...")
    print("Open http://localhost:5000 in your browser")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
