"""
Unit tests for ennoia_client_lic.py
Tests the license client functionality.
"""
import pytest
import sys
import os
import json
import hashlib
import uuid
from unittest.mock import Mock, patch, mock_open
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ennoia_client_lic import get_fingerprint, request_license, verify_license_file, SERVER_URL


class TestGetFingerprint:
    """Tests for the get_fingerprint function."""

    def test_fingerprint_returns_string(self):
        """Test that fingerprint returns a string."""
        fp = get_fingerprint()
        assert isinstance(fp, str)

    def test_fingerprint_is_hex(self):
        """Test that fingerprint is a valid hex string."""
        fp = get_fingerprint()
        # SHA256 produces 64 hex characters
        assert len(fp) == 64
        assert all(c in '0123456789abcdef' for c in fp)

    def test_fingerprint_is_consistent(self):
        """Test that fingerprint is consistent across calls."""
        fp1 = get_fingerprint()
        fp2 = get_fingerprint()
        assert fp1 == fp2

    def test_fingerprint_uses_mac_address(self):
        """Test that fingerprint is based on MAC address."""
        mac = uuid.getnode()
        expected = hashlib.sha256(f"{mac}".encode()).hexdigest()
        assert get_fingerprint() == expected


class TestRequestLicense:
    """Tests for the request_license function."""

    @patch('ennoia_client_lic.requests.post')
    def test_request_license_success(self, mock_post, tmp_path):
        """Test successful license request."""
        mock_response = Mock()
        mock_response.json.return_value = {"license": "test", "signature": "sig"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        with patch('builtins.open', mock_open()) as mock_file:
            request_license("TEST-KEY-123")

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == f"{SERVER_URL}/verify"
        assert "license_key" in call_args[1]["json"]
        assert call_args[1]["json"]["license_key"] == "TEST-KEY-123"

    @patch('ennoia_client_lic.requests.post')
    def test_request_license_includes_fingerprint(self, mock_post):
        """Test that license request includes machine fingerprint."""
        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        with patch('builtins.open', mock_open()):
            request_license("KEY")

        call_args = mock_post.call_args
        assert "fingerprint" in call_args[1]["json"]
        assert call_args[1]["json"]["fingerprint"] == get_fingerprint()

    @patch('ennoia_client_lic.requests.post')
    def test_request_license_handles_error(self, mock_post, capsys):
        """Test that license request handles errors gracefully."""
        mock_post.side_effect = Exception("Network error")

        request_license("KEY")

        captured = capsys.readouterr()
        assert "failed" in captured.out.lower() or "❌" in captured.out


class TestVerifyLicenseFile:
    """Tests for the verify_license_file function."""

    def test_verify_missing_file(self, capsys):
        """Test verification with missing license file."""
        with patch('builtins.open', side_effect=FileNotFoundError()):
            result = verify_license_file()

        assert result == 0
        captured = capsys.readouterr()
        assert "failed" in captured.out.lower() or "❌" in captured.out

    @patch('ennoia_client_lic.requests.get')
    def test_verify_expired_license(self, mock_get, capsys):
        """Test verification with expired license."""
        expired_date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
        license_data = {
            "license": {
                "expires": expired_date,
                "fingerprint": get_fingerprint()
            },
            "signature": "dGVzdA=="  # base64 encoded "test"
        }

        with patch('builtins.open', mock_open(read_data=json.dumps(license_data))):
            result = verify_license_file()

        assert result == 0
        captured = capsys.readouterr()
        assert "expired" in captured.out.lower()

    @patch('ennoia_client_lic.requests.get')
    def test_verify_wrong_fingerprint(self, mock_get, capsys):
        """Test verification with wrong machine fingerprint."""
        future_date = (datetime.utcnow() + timedelta(days=30)).strftime("%Y-%m-%d")
        license_data = {
            "license": {
                "expires": future_date,
                "fingerprint": "wrong_fingerprint_hash"
            },
            "signature": "dGVzdA=="
        }

        with patch('builtins.open', mock_open(read_data=json.dumps(license_data))):
            result = verify_license_file()

        assert result == 0
        captured = capsys.readouterr()
        assert "not valid for this machine" in captured.out.lower() or "fingerprint" in captured.out.lower()

    def test_verify_invalid_json(self, capsys):
        """Test verification with invalid JSON."""
        with patch('builtins.open', mock_open(read_data="not valid json")):
            result = verify_license_file()

        assert result == 0


class TestServerURL:
    """Tests for server configuration."""

    def test_server_url_defined(self):
        """Test that SERVER_URL is defined."""
        assert SERVER_URL is not None
        assert isinstance(SERVER_URL, str)
        assert SERVER_URL.startswith("http")
