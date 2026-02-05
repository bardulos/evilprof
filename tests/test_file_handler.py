# tests/test_file_handler.py
import pytest
import pandas as pd
import io
import os

from config import QUESTION_TYPE_MC, QUESTION_TYPE_OE, QUESTION_TYPE_UNKNOWN


class FakeUploadedFile:
    """Mimics a Streamlit UploadedFile for testing."""
    def __init__(self, name, content_bytes):
        self.name = name
        self._content = content_bytes
        self._pos = 0

    def getvalue(self):
        return self._content

    def seek(self, pos):
        self._pos = pos

    def read(self, n=-1):
        if n == -1:
            data = self._content[self._pos:]
            self._pos = len(self._content)
        else:
            data = self._content[self._pos:self._pos + n]
            self._pos += n
        return data


class TestLoadQuestionsFromExcel:
    def _collect_callback(self):
        messages = []
        def callback(msg_type, msg_key, **kwargs):
            messages.append((msg_type, msg_key, kwargs))
        return callback, messages

    def test_returns_error_for_none_file(self):
        from file_handler import load_questions_from_excel
        callback, messages = self._collect_callback()
        result = load_questions_from_excel(None, callback)
        assert result == (None, None, "UPLOAD_FIRST_WARNING")

    def test_returns_error_for_unsupported_format(self):
        from file_handler import load_questions_from_excel
        callback, messages = self._collect_callback()
        fake_file = FakeUploadedFile("test.txt", b"hello")
        result = load_questions_from_excel(fake_file, callback)
        assert result[2] == "FH_UNSUPPORTED_FORMAT"

    def test_loads_csv_with_mc_questions(self):
        from file_handler import load_questions_from_excel
        callback, messages = self._collect_callback()
        csv_content = "Q1,A,B,C\nQ2,D,E,F\n"
        fake_file = FakeUploadedFile("test.csv", csv_content.encode('utf-8'))
        all_q, blocks_sum, error = load_questions_from_excel(fake_file, callback)
        assert error is None
        assert len(all_q) == 2
        assert all_q[0]['type'] == QUESTION_TYPE_MC
        assert len(blocks_sum) == 1
        assert blocks_sum[0]['type'] == QUESTION_TYPE_MC

    def test_loads_csv_with_open_ended_questions(self):
        from file_handler import load_questions_from_excel
        callback, messages = self._collect_callback()
        csv_content = "Q1\nQ2\n"
        fake_file = FakeUploadedFile("test.csv", csv_content.encode('utf-8'))
        all_q, blocks_sum, error = load_questions_from_excel(fake_file, callback)
        assert error is None
        assert len(all_q) == 2
        assert all_q[0]['type'] == QUESTION_TYPE_OE

    def test_detects_multiple_blocks(self):
        from file_handler import load_questions_from_excel
        callback, messages = self._collect_callback()
        # Use comma-separated empty fields to create a truly empty row for pandas
        csv_content = "Q1,A,B,C\nQ2,D,E,F\n,,,\nQ3,,,\nQ4,,,\n"
        fake_file = FakeUploadedFile("test.csv", csv_content.encode('utf-8'))
        all_q, blocks_sum, error = load_questions_from_excel(fake_file, callback)
        assert error is None
        assert len(blocks_sum) == 2
        assert blocks_sum[0]['type'] == QUESTION_TYPE_MC
        assert blocks_sum[1]['type'] == QUESTION_TYPE_OE

    def test_warns_on_mixed_types_in_block(self):
        from file_handler import load_questions_from_excel
        callback, messages = self._collect_callback()
        # First question MC (3 answers), second question OE (no answers) => mixed
        csv_content = "Q1,A,B,C\nQ2\n"
        fake_file = FakeUploadedFile("test.csv", csv_content.encode('utf-8'))
        all_q, blocks_sum, error = load_questions_from_excel(fake_file, callback)
        assert error is None
        # Q2 should be skipped due to mixed type warning
        assert len(all_q) == 1
        warning_messages = [m for m in messages if m[0] == 'warning']
        assert len(warning_messages) >= 1

    def test_empty_file_returns_error(self):
        from file_handler import load_questions_from_excel
        callback, messages = self._collect_callback()
        csv_content = "\n\n\n"
        fake_file = FakeUploadedFile("test.csv", csv_content.encode('utf-8'))
        all_q, blocks_sum, error = load_questions_from_excel(fake_file, callback)
        assert error is not None
