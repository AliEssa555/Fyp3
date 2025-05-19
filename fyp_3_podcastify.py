import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from podcastfy.client import generate_podcast
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration
AUDIO_DIR = "generated_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Ensure required environment variables are set
if not os.getenv("GROQ_API_KEY"):
    logger.warning("GROQ_API_KEY not found in environment variables")
    os.environ["GROQ_API_KEY"] = "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"

os.environ["ELEVENLABS_API_KEY"] = "sk_86cdec0196d26a4dd7cfeed43d67adb812ae3e7d879dc6e2"

@app.route('/')
def home():
    return '''
    <h1>Podcastify YouTube to Podcast</h1>
    <form action="/generate" method="post">
        <label for="youtube_url">YouTube URL:</label><br>
        <input type="text" id="youtube_url" name="youtube_url" required><br><br>
        <input type="submit" value="Generate Podcast">
    </form>
    '''

@app.route('/generate', methods=['POST'])
def generate():
    youtube_url = request.form.get('youtube_url')
    print(youtube_url)
    if not youtube_url:
        return jsonify({'error': 'YouTube URL is required'}), 400
    
    try:
        audio_file = generate_podcast(urls=[youtube_url] ,llm_model_name="groq/gemma2-9b-it",api_key_label="GROQ_API_KEY",tts_model="elevenlabs",longform=True)
        
        if not audio_file:
            raise ValueError("No audio file path returned from generate_podcast")
            
        logger.info(f"Generated audio file at: {audio_file}")

        # Handle file movement
        audio_file_name = os.path.basename(audio_file)
        static_audio_path = os.path.join(AUDIO_DIR, audio_file_name)
        
        # Use copy instead of rename to preserve original
        with open(audio_file, 'rb') as src, open(static_audio_path, 'wb') as dst:
            dst.write(src.read())
        
        # Return response with audio player
        return f'''
        <h2>Podcast Generated!</h2>
        <p>Audio file: {audio_file_name}</p>
        <audio controls>
            <source src="/audio/{audio_file_name}" type="audio/mpeg">
            Your browser does not support the audio element.
        </audio>
        <br>
        <a href="/">Generate another podcast</a>
        '''
        
    except Exception as e:
        logger.error(f"Error generating podcast: {str(e)}")
        return jsonify({
            'error': 'Failed to generate podcast',
            'message': str(e)
        }), 500

@app.route('/audio/<filename>')
def serve_audio(filename):
    try:
        return send_from_directory(AUDIO_DIR, filename)
    except FileNotFoundError:
        return jsonify({'error': 'Audio file not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
