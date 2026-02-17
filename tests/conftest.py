import pytest
import json
import os
import tempfile
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from tests.test_helpers import setup_streamlit_mocks

patches = setup_streamlit_mocks()
for key, value in patches.items():
    patch(key, value).start()


@pytest.fixture(scope="session", autouse=True)
def setup_mocks():
    yield
    for p in patches.values():
        if hasattr(p, 'stop'):
            p.stop()


@pytest.fixture
def sample_dataset():
    return [
        {
            "question": "What is discipline?",
            "answer": "Discipline is the practice of training oneself to follow rules or a code of behavior."
        },
        {
            "question": "How to maintain life balance?",
            "answer": "Life balance involves managing time between work, personal life, health, and relationships."
        },
        {
            "question": "What is mentorship?",
            "answer": "Mentorship is a relationship where an experienced person guides and supports a less experienced person."
        }
    ]


@pytest.fixture
def temp_dataset_file(sample_dataset):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_dataset, f)
        temp_path = f.name
    
    yield temp_path
    
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def mock_api_key():
    return "test-api-key-12345"


@pytest.fixture
def empty_dataset():
    return []
