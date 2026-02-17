import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import (
    _norm,
    retrieve_context,
    pick_default_model,
    build_prompt,
    text_to_speech,
    detect_audio_mime
)


class TestNormalization:
    def test_norm_basic(self):
        assert _norm("Hello World") == "hello world"
    
    def test_norm_whitespace(self):
        assert _norm("  Hello   World  ") == "hello world"
    
    def test_norm_empty(self):
        assert _norm("") == ""
        assert _norm(None) == ""
    
    def test_norm_case_insensitive(self):
        assert _norm("HELLO WORLD") == "hello world"
        assert _norm("HeLlO WoRlD") == "hello world"


class TestContextRetrieval:
    def test_retrieve_context_exact_match(self, sample_dataset):
        query = "What is discipline?"
        result = retrieve_context(query, sample_dataset, k=1)
        assert "discipline" in result.lower()
        assert "training" in result.lower()
    
    def test_retrieve_context_partial_match(self, sample_dataset):
        query = "life balance"
        result = retrieve_context(query, sample_dataset, k=1)
        assert "balance" in result.lower() or "life" in result.lower()
    
    def test_retrieve_context_no_match(self, sample_dataset):
        query = "completely unrelated topic"
        result = retrieve_context(query, sample_dataset, k=1)
        assert isinstance(result, str)
    
    def test_retrieve_context_empty_query(self, sample_dataset):
        result = retrieve_context("", sample_dataset, k=1)
        assert result == ""
    
    def test_retrieve_context_empty_dataset(self):
        result = retrieve_context("test query", [], k=1)
        assert result == ""
    
    def test_retrieve_context_k_parameter(self, sample_dataset):
        query = "discipline"
        result = retrieve_context(query, sample_dataset, k=2)
        assert result.count("[") == 2


class TestModelSelection:
    def test_pick_default_model_preferred_exists(self):
        available = ["models/gemini-1.5-flash", "models/other"]
        result = pick_default_model(available)
        assert result == "models/gemini-1.5-flash"
    
    def test_pick_default_model_fallback(self):
        available = ["models/other-model"]
        result = pick_default_model(available)
        assert result == "models/other-model"
    
    def test_pick_default_model_empty_list(self):
        result = pick_default_model([])
        assert result == "models/gemini-pro"


class TestPromptBuilding:
    def test_build_prompt_basic(self):
        user_input = "What is discipline?"
        context = "Test context"
        history = []
        result = build_prompt(user_input, context, history)
        assert "What is discipline?" in result
        assert "Test context" in result
        assert "Student:" in result
        assert "Mentor:" in result
    
    def test_build_prompt_with_history(self):
        user_input = "Follow up question"
        context = "Context"
        history = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"}
        ]
        result = build_prompt(user_input, context, history)
        assert "First question" in result
        assert "First answer" in result
        assert "Follow up question" in result
    
    def test_build_prompt_empty_context(self):
        user_input = "Question"
        context = ""
        history = []
        result = build_prompt(user_input, context, history)
        assert "No additional context available" in result
    
    def test_build_prompt_history_limit(self):
        user_input = "Question"
        context = "Context"
        history = [{"role": "user", "content": f"Question {i}"} for i in range(20)]
        result = build_prompt(user_input, context, history)
        assert len(history) == 20
        assert "Question 19" in result


class TestTextToSpeech:
    @patch('app.gTTS')
    def test_text_to_speech_success(self, mock_gtts):
        mock_tts_instance = MagicMock()
        mock_gtts.return_value = mock_tts_instance
        
        from io import BytesIO
        result = text_to_speech("Hello world")
        
        assert result is not None
        mock_gtts.assert_called_once()
        mock_tts_instance.write_to_fp.assert_called_once()
    
    def test_text_to_speech_empty_text(self):
        result = text_to_speech("")
        assert result is None
    
    def test_text_to_speech_whitespace_only(self):
        result = text_to_speech("   ")
        assert result is None
    
    @patch('app.gTTS')
    def test_text_to_speech_error_handling(self, mock_gtts):
        mock_gtts.side_effect = Exception("TTS Error")
        result = text_to_speech("Test")
        assert result is None


class TestAudioMimeDetection:
    def test_detect_audio_mime_wav(self):
        wav_header = b"RIFF" + b"\x00" * 4 + b"WAVE"
        result = detect_audio_mime(wav_header)
        assert result == "audio/wav"
    
    def test_detect_audio_mime_ogg(self):
        ogg_header = b"OggS"
        result = detect_audio_mime(ogg_header)
        assert result == "audio/ogg"
    
    def test_detect_audio_mime_webm(self):
        webm_header = b"\x1a\x45\xdf\xa3"
        result = detect_audio_mime(webm_header)
        assert result == "audio/webm"
    
    def test_detect_audio_mime_mp3(self):
        mp3_header = b"ID3"
        result = detect_audio_mime(mp3_header)
        assert result == "audio/mpeg"
    
    def test_detect_audio_mime_flac(self):
        flac_header = b"fLaC"
        result = detect_audio_mime(flac_header)
        assert result == "audio/flac"
    
    def test_detect_audio_mime_empty(self):
        result = detect_audio_mime(b"")
        assert result == "audio/wav"
    
    def test_detect_audio_mime_unknown(self):
        unknown = b"unknown format"
        result = detect_audio_mime(unknown)
        assert result == "application/octet-stream"
