import os
import sys
import json
import argparse
import tempfile
import traceback
from datetime import datetime  # Added this import
from vosk import Model, KaldiRecognizer
import wave
from pydub import AudioSegment
from phonemizer import phonemize
from difflib import SequenceMatcher

# Configuration
DEBUG = True
MIN_AUDIO_LENGTH_MS = 500
SAMPLE_RATE = 16000

# Supported languages
SUPPORTED_LANGUAGES = {
    'en': 'en-us',
    'fr': 'fr',
    'de': 'de',
    'nl': 'nl',
    'ko': 'ko'
}

LANGUAGE_TIPS = {
    'fr': ["Nasal vowels...", "Final consonants...", "French 'r'..."],
    'de': ["Consonants clearly...", "Vowel length...", "'ch' sounds..."],
    'nl': ["Guttural 'g'...", "Vowel length...", "'ui' diphthong..."],
    'ko': ["Tense consonants...", "Vowel combinations...", "Even syllables..."],
    'en': ["Stress syllables...", "Vowel sounds...", "Final consonants..."]
}

def debug_log(message):
    if DEBUG:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] {message}")

def validate_audio(audio_path):
    debug_log(f"Validating audio file: {audio_path}")
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    try:
        audio = AudioSegment.from_file(audio_path)
        debug_log(f"Audio info - Duration: {len(audio)}ms, Channels: {audio.channels}, Sample rate: {audio.frame_rate}Hz")
        
        if len(audio) < MIN_AUDIO_LENGTH_MS:
            raise ValueError(f"Audio too short (minimum {MIN_AUDIO_LENGTH_MS}ms required)")
            
        return audio
    except Exception as e:
        raise ValueError(f"Audio validation failed: {str(e)}")

def convert_audio(audio):
    try:
        debug_log("Converting audio to 16kHz mono...")
        return audio.set_channels(1).set_frame_rate(SAMPLE_RATE)
    except Exception as e:
        raise ValueError(f"Audio conversion failed: {str(e)}")

def load_model(language):
    lang_code = language if language in SUPPORTED_LANGUAGES else 'en'
    model_dir = SUPPORTED_LANGUAGES[lang_code]
    model_path = os.path.join("vosk_models", model_dir)
    
    debug_log(f"Loading model for {language} from: {model_path}")
    
    if not os.path.exists(model_path):
        raise ValueError(f"Model directory not found: {model_path}")
    if not os.path.exists(os.path.join(model_path, "am", "final.mdl")):
        raise ValueError("Model files are incomplete")
    
    try:
        return Model(model_path)
    except Exception as e:
        raise ValueError(f"Model loading failed: {str(e)}")

def recognize_speech(audio, model):
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
            temp_path = tmp_wav.name
            debug_log(f"Creating temporary WAV at: {temp_path}")
            audio.export(temp_path, format="wav")
            
            recognizer = KaldiRecognizer(model, SAMPLE_RATE)
            recognizer.SetWords(True)
            
            results = []
            with wave.open(temp_path, 'rb') as wf:
                debug_log("Starting speech recognition...")
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    if recognizer.AcceptWaveform(data):
                        result = json.loads(recognizer.Result())
                        results.append(result.get('text', ''))
                
                final = json.loads(recognizer.FinalResult())
                if final.get('text'):
                    results.append(final.get('text', ''))
            
            recognized_text = " ".join(results).strip()
            debug_log(f"Recognized text: '{recognized_text}'")
            
            return recognized_text
            
    except Exception as e:
        raise ValueError(f"Speech recognition failed: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            debug_log("Removed temporary WAV file")

def get_reference_text(recognized_text, lang_code):
    COMMON_PHRASES = {
        'fr': ["bonjour", "comment ça va", "je m'appelle"],
        'de': ["hallo", "wie geht's", "mein name ist"],
        'nl': ["hallo", "hoe gaat het", "mijn naam is"],
        'ko': ["안녕하세요", "이름이 뭐예요", "감사합니다"],
        'en': ["hello", "how are you", "my name is"]
    }
    
    words = recognized_text.lower().split()
    for phrase in COMMON_PHRASES.get(lang_code, COMMON_PHRASES['en']):
        phrase_words = phrase.split()
        if SequenceMatcher(None, words, phrase_words).ratio() > 0.7:
            return phrase
    return recognized_text

def detect_mistakes(recognized_text, reference_text, lang_code):
    recognized_words = recognized_text.lower().split()
    reference_words = reference_text.lower().split()
    
    mistakes = []
    corrected_words = []
    matcher = SequenceMatcher(None, recognized_words, reference_words)
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ('replace', 'delete'):
            for i in range(i1, i2):
                if i < len(recognized_words):
                    mistake = {
                        "word": recognized_words[i],
                        "correct": reference_words[j1] if j1 < len(reference_words) else "",
                        "position": i,
                        "type": "incorrect_word"
                    }
                    mistakes.append(mistake)
                    corrected_words.append(mistake["correct"] if mistake["correct"] else "[silence]")
        
        elif tag == 'insert':
            for i in range(i1, i2):
                mistakes.append({
                    "word": recognized_words[i],
                    "correct": "",
                    "position": i,
                    "type": "extra_word"
                })
        
        elif tag == 'equal':
            corrected_words.extend(recognized_words[i1:i2])
    
    if lang_code in ['en', 'fr', 'de', 'nl', 'ko']:
        try:
            recognized_phonemes = phonemize(recognized_text, language=lang_code, 
                                          backend='espeak', strip=True, preserve_punctuation=False)
            reference_phonemes = phonemize(reference_text, language=lang_code,
                                         backend='espeak', strip=True, preserve_punctuation=False)
            
            if SequenceMatcher(None, recognized_phonemes.split(), reference_phonemes.split()).ratio() < 0.85:
                for mistake in mistakes:
                    if mistake["type"] == "incorrect_word":
                        mistake["type"] = "pronunciation"
        except Exception as e:
            debug_log(f"Phonemization failed: {str(e)}")
    
    return mistakes, " ".join(corrected_words)

def generate_feedback(mistakes, lang_code):
    feedback = []
    
    if not mistakes:
        feedback.append("Excellent pronunciation! No mistakes detected.")
        feedback.extend(LANGUAGE_TIPS.get(lang_code, LANGUAGE_TIPS['en'])[:2])
        return feedback
    
    pronunciation_errors = [m for m in mistakes if m["type"] == "pronunciation"]
    word_errors = [m for m in mistakes if m["type"] == "incorrect_word"]
    extra_words = [m for m in mistakes if m["type"] == "extra_word"]
    
    if pronunciation_errors:
        feedback.append(f"Pronunciation issues ({len(pronunciation_errors)}):")
        for err in pronunciation_errors[:3]:
            feedback.append(f"• '{err['word']}' sounded like {err['word']} instead of {err['correct']}")
    
    if word_errors:
        feedback.append(f"Incorrect words ({len(word_errors)}):")
        for err in word_errors[:3]:
            if err["correct"]:
                feedback.append(f"• Used '{err['word']}' instead of '{err['correct']}'")
            else:
                feedback.append(f"• Unrecognized word: '{err['word']}'")
    
    if extra_words:
        feedback.append(f"Extra words detected ({len(extra_words)})")
    
    feedback.append("\nTips for improvement:")
    feedback.extend(LANGUAGE_TIPS.get(lang_code, LANGUAGE_TIPS['en']))
    
    return feedback

def calculate_score(recognized_text, reference_text, mistakes):
    if not reference_text:
        return 0
    
    similarity = SequenceMatcher(None, recognized_text.lower(), reference_text.lower()).ratio()
    base_score = similarity * 100
    
    penalty = sum(
        3 if m["type"] == "pronunciation" else 
        2 if m["type"] == "incorrect_word" else 
        1 for m in mistakes
    )
    
    return max(0, min(100, base_score - min(penalty, 40)))

def analyze_pronunciation(audio_path, language='en'):
    try:
        debug_log(f"\n{' Starting Analysis ':=^50}")
        debug_log(f"Language: {language}")
        debug_log(f"Audio path: {audio_path}")
        
        audio = validate_audio(audio_path)
        processed_audio = convert_audio(audio)
        model = load_model(language)
        
        recognized_text = recognize_speech(processed_audio, model)
        
        if not recognized_text:
            return {
                "error": "No speech detected",
                "debug": {
                    "duration_ms": len(processed_audio),
                    "sample_rate": processed_audio.frame_rate,
                    "channels": processed_audio.channels
                }
            }
        
        reference_text = get_reference_text(recognized_text, language)
        mistakes, corrected_text = detect_mistakes(recognized_text, reference_text, language)
        feedback = generate_feedback(mistakes, language)
        score = calculate_score(recognized_text, reference_text, mistakes)
        
        debug_log(f"{' Analysis Complete ':=^50}")
        debug_log(f"Score: {score}")
        
        return {
            "original_text": recognized_text,
            "reference_text": reference_text,
            "corrected_text": corrected_text,
            "mistakes": mistakes,
            "feedback": feedback,
            "score": score,
            "language": language
        }
        
    except Exception as e:
        error_info = {
            "error": "Analysis failed",
            "message": str(e),
            "type": type(e).__name__,
            "timestamp": datetime.now().isoformat()
        }
        
        if DEBUG:
            error_info["traceback"] = traceback.format_exc()
            debug_log(f"{' ERROR ':=^50}")
            debug_log(json.dumps(error_info, indent=2))
        
        return error_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APMD Pronunciation Analyzer")
    parser.add_argument('--audio', required=True, help="Path to audio file")
    parser.add_argument('--language', default='en', choices=SUPPORTED_LANGUAGES.keys())
    parser.add_argument('--output', help="Output JSON file path")
    
    args = parser.parse_args()
    
    result = analyze_pronunciation(args.audio, args.language)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        debug_log(f"Results saved to {args.output}")
    else:
        print(json.dumps(result, indent=2))
