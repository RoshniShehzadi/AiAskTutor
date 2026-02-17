import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock, mock_open
import json
import os

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDatasetLoading:
    def test_load_dataset_success(self, sample_dataset):
        with patch('builtins.open', mock_open(read_data=json.dumps(sample_dataset))):
            with patch('json.load', return_value=sample_dataset):
                with patch('app.st.cache_data', lambda x: x):
                    from app import load_dataset
                    result = load_dataset()
        
        assert isinstance(result, list)
        assert len(result) > 0
    
    def test_load_dataset_file_not_found(self):
        with patch('builtins.open', side_effect=FileNotFoundError):
            with patch('app.st.cache_data', lambda x: x):
                with patch('app.st.error') as mock_error:
                    from app import load_dataset
                    result = load_dataset()
        
        assert result == []
        mock_error.assert_called()
    
    def test_load_dataset_invalid_json(self):
        with patch('builtins.open', mock_open(read_data="invalid json")):
            with patch('json.load', side_effect=json.JSONDecodeError("", "", 0)):
                with patch('app.st.cache_data', lambda x: x):
                    with patch('app.st.error') as mock_error:
                        from app import load_dataset
                        result = load_dataset()
        
        assert result == []
        mock_error.assert_called()


class TestAnswerGeneration:
    @patch('app.genai.GenerativeModel')
    def test_generate_answer_success(self, mock_model_class):
        with patch('app.st.error'):
            from app import generate_answer
        
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is a test answer about discipline."
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        with patch('app.MODEL_NAME', 'models/gemini-pro'):
            result = generate_answer("What is discipline?", "context", [])
        
        assert result == "This is a test answer about discipline."
        mock_model.generate_content.assert_called_once()
    
    @patch('app.genai.GenerativeModel')
    @patch('app.st.error')
    def test_generate_answer_api_error(self, mock_error, mock_model_class):
        from app import generate_answer
        
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("API Error")
        mock_model_class.return_value = mock_model
        
        with patch('app.MODEL_NAME', 'models/gemini-pro'):
            result = generate_answer("Question", "context", [])
        
        assert result == ""
        mock_error.assert_called()
    
    @patch('app.genai.GenerativeModel')
    def test_generate_answer_empty_response(self, mock_model_class):
        with patch('app.st.error'):
            from app import generate_answer
        
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = None
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        with patch('app.MODEL_NAME', 'models/gemini-pro'):
            result = generate_answer("Question", "context", [])
        
        assert result == ""


class TestAudioTranscription:
    @patch('app.genai.GenerativeModel')
    def test_transcribe_audio_success(self, mock_model_class):
        with patch('app.st.error'):
            from app import transcribe_audio
        
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is transcribed text"
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        with patch('app.MODEL_NAME', 'models/gemini-pro'):
            result = transcribe_audio(b"fake audio bytes", "audio/wav")
        
        assert result == "This is transcribed text"
        mock_model.generate_content.assert_called_once()
    
    @patch('app.genai.GenerativeModel')
    @patch('app.st.error')
    def test_transcribe_audio_error(self, mock_error, mock_model_class):
        from app import transcribe_audio
        
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("Transcription error")
        mock_model_class.return_value = mock_model
        
        with patch('app.MODEL_NAME', 'models/gemini-pro'):
            result = transcribe_audio(b"fake audio bytes", "audio/wav")
        
        assert result == ""
        mock_error.assert_called()


class TestEndToEndFlow:
    @patch('app.genai.configure')
    @patch('app.genai.GenerativeModel')
    @patch('app.load_dataset')
    @patch('app.st.chat_input')
    @patch('app.st.rerun')
    def test_complete_chat_flow(self, mock_rerun, mock_chat_input, mock_load_dataset, 
                                mock_model_class, mock_configure, sample_dataset):
        with patch('app.st.error'):
            from app import retrieve_context, generate_answer
        
        mock_load_dataset.return_value = sample_dataset
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Discipline is important for personal growth."
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        query = "What is discipline?"
        context = retrieve_context(query, sample_dataset, k=3)
        
        with patch('app.MODEL_NAME', 'models/gemini-pro'):
            answer = generate_answer(query, context, [])
        
        assert context != ""
        assert answer != ""
        assert "discipline" in answer.lower()
    
    @patch('app.genai.configure')
    @patch('app.genai.GenerativeModel')
    @patch('app.load_dataset')
    def test_context_retrieval_and_answer_integration(self, mock_load_dataset, 
                                                      mock_model_class, mock_configure, 
                                                      sample_dataset):
        with patch('app.st.error'):
            from app import retrieve_context, generate_answer, build_prompt
        
        mock_load_dataset.return_value = sample_dataset
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Life balance requires managing priorities."
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        query = "How to maintain life balance?"
        context = retrieve_context(query, sample_dataset, k=2)
        prompt = build_prompt(query, context, [])
        
        with patch('app.MODEL_NAME', 'models/gemini-pro'):
            answer = generate_answer(query, context, [])
        
        assert "balance" in context.lower() or "life" in context.lower()
        assert query in prompt
        assert answer != ""
    
    @patch('app.gTTS')
    def test_audio_generation_for_response(self, mock_gtts):
        from app import text_to_speech
        
        mock_tts_instance = MagicMock()
        mock_gtts.return_value = mock_tts_instance
        
        response_text = "This is a meaningful response that should be converted to speech."
        audio_result = text_to_speech(response_text)
        
        assert audio_result is not None
        mock_gtts.assert_called_once()
    
    def test_empty_response_no_audio(self):
        from app import text_to_speech
        
        result = text_to_speech("")
        assert result is None
        
        result = text_to_speech("   ")
        assert result is None


class TestErrorHandling:
    def test_missing_api_key_handling(self):
        with patch('app.st.text_input', return_value=""):
            with patch('app.st.stop') as mock_stop:
                with patch('app.st.warning'):
                    try:
                        import importlib
                        if 'app' in sys.modules:
                            importlib.reload(sys.modules['app'])
                        else:
                            import app
                    except (SystemExit, KeyError):
                        pass
                    mock_stop.assert_called()
    
    @patch('app.genai.GenerativeModel')
    @patch('app.st.error')
    def test_model_error_handling(self, mock_error, mock_model_class):
        from app import generate_answer
        
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("Model not found")
        mock_model_class.return_value = mock_model
        
        with patch('app.MODEL_NAME', 'models/invalid-model'):
            result = generate_answer("test", "context", [])
        
        assert result == ""
        mock_error.assert_called()
    
    def test_dataset_loading_error_handling(self):
        with patch('builtins.open', side_effect=Exception("File read error")):
            with patch('app.st.cache_data', lambda x: x):
                with patch('app.st.error') as mock_error:
                    from app import load_dataset
                    result = load_dataset()
        
        assert result == []
        mock_error.assert_called()
