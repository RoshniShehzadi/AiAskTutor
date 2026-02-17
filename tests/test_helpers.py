import sys
from unittest.mock import MagicMock
from pathlib import Path

def setup_streamlit_mocks():
    mock_session_state = MagicMock()
    mock_session_state.__contains__ = MagicMock(return_value=False)
    mock_session_state.__getitem__ = MagicMock(side_effect=KeyError("Key not found"))
    mock_session_state.__setitem__ = MagicMock()
    mock_session_state.get = MagicMock(return_value=None)
    
    patches = {
        'streamlit.session_state': mock_session_state,
        'streamlit.set_page_config': MagicMock(),
        'streamlit.sidebar': MagicMock(),
        'streamlit.header': MagicMock(),
        'streamlit.caption': MagicMock(),
        'streamlit.columns': MagicMock(return_value=[MagicMock(), MagicMock()]),
        'streamlit.write': MagicMock(),
        'streamlit.chat_input': MagicMock(return_value=None),
        'streamlit.chat_message': MagicMock(),
        'streamlit.spinner': MagicMock(),
        'streamlit.rerun': MagicMock(),
        'streamlit.error': MagicMock(),
        'streamlit.warning': MagicMock(),
        'streamlit.stop': MagicMock(),
        'streamlit.image': MagicMock(),
        'streamlit.title': MagicMock(),
        'streamlit.markdown': MagicMock(),
        'streamlit.text_input': MagicMock(return_value="test-key"),
        'streamlit.selectbox': MagicMock(return_value="models/gemini-pro"),
        'streamlit.audio': MagicMock(),
        'streamlit.cache_data': lambda **kwargs: lambda func: func,
    }
    
    return patches
