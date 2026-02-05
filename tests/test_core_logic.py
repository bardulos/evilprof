# tests/test_core_logic.py
import random
import pytest
from core_logic import weighted_random_sample_without_replacement, generate_all_tests_data
from config import QUESTION_TYPE_MC, QUESTION_TYPE_OE


# ================================================================
# Tests for weighted_random_sample_without_replacement
# ================================================================

class TestWRSwOR:
    def test_returns_empty_for_k_zero(self):
        result = weighted_random_sample_without_replacement([1, 2, 3], [1, 1, 1], 0)
        assert result == []

    def test_returns_all_when_k_equals_n(self):
        population = [10, 20, 30]
        result = weighted_random_sample_without_replacement(population, [1, 1, 1], 3)
        assert sorted(result) == sorted(population)

    def test_returns_all_when_k_exceeds_n(self):
        population = [10, 20]
        result = weighted_random_sample_without_replacement(population, [1, 1], 5)
        assert sorted(result) == sorted(population)

    def test_returns_correct_count(self):
        population = list(range(20))
        weights = [1] * 20
        result = weighted_random_sample_without_replacement(population, weights, 5)
        assert len(result) == 5

    def test_no_duplicates(self):
        population = list(range(50))
        weights = [1] * 50
        result = weighted_random_sample_without_replacement(population, weights, 10)
        assert len(result) == len(set(result))

    def test_all_results_from_population(self):
        population = list(range(10))
        weights = [1] * 10
        result = weighted_random_sample_without_replacement(population, weights, 3)
        for item in result:
            assert item in population

    def test_raises_on_negative_k(self):
        with pytest.raises(ValueError):
            weighted_random_sample_without_replacement([1, 2], [1, 1], -1)

    def test_raises_on_mismatched_lengths(self):
        with pytest.raises(ValueError):
            weighted_random_sample_without_replacement([1, 2, 3], [1, 1], 2)

    def test_skips_zero_weight_elements(self):
        population = [10, 20, 30, 40]
        weights = [0, 1, 1, 0]
        result = weighted_random_sample_without_replacement(population, weights, 2)
        assert sorted(result) == [20, 30]

    def test_raises_when_not_enough_positive_weights(self):
        population = [10, 20, 30]
        weights = [0, 1, 0]
        with pytest.raises(ValueError):
            weighted_random_sample_without_replacement(population, weights, 2)

    def test_higher_weights_favored_statistically(self):
        random.seed(42)
        population = [0, 1]
        weights = [100, 1]
        counts = {0: 0, 1: 0}
        for _ in range(1000):
            result = weighted_random_sample_without_replacement(population, weights, 1)
            counts[result[0]] += 1
        assert counts[0] > counts[1]


# ================================================================
# Tests for generate_all_tests_data
# ================================================================

def _make_questions(block_id, q_type, count, start_index=0):
    """Helper to create a list of question dicts for testing."""
    questions = []
    for i in range(count):
        q = {
            'question': f"Block {block_id} Q{i+1}",
            'answers': [f"A{j}" for j in range(4)] if q_type == QUESTION_TYPE_MC else [],
            'original_index': start_index + i,
            'type': q_type,
            'block_id': block_id,
        }
        questions.append(q)
    return questions


class TestGenerateAllTestsData:
    def _nop_callback(self, *args, **kwargs):
        pass

    def test_generates_correct_number_of_tests(self):
        questions = _make_questions(1, QUESTION_TYPE_MC, 12)
        block_requests = {1: 3}
        tests, messages = generate_all_tests_data(questions, block_requests, 5, self._nop_callback)
        assert len(tests) == 5

    def test_each_test_has_correct_question_count(self):
        questions = _make_questions(1, QUESTION_TYPE_MC, 12)
        block_requests = {1: 4}
        tests, messages = generate_all_tests_data(questions, block_requests, 10, self._nop_callback)
        for test in tests:
            assert len(test) == 4

    def test_wrswor_no_overlap_consecutive(self):
        """When n > 2k, consecutive tests should have no overlapping questions."""
        random.seed(123)
        questions = _make_questions(1, QUESTION_TYPE_OE, 12)
        block_requests = {1: 4}
        tests, messages = generate_all_tests_data(questions, block_requests, 10, self._nop_callback)
        for i in range(len(tests) - 1):
            set_a = set(q['original_index'] for q in tests[i])
            set_b = set(q['original_index'] for q in tests[i + 1])
            assert set_a.isdisjoint(set_b), f"Tests {i} and {i+1} share questions"

    def test_simple_random_when_k_half_of_n(self):
        """When k >= n/2, uses simple random. Should still produce valid tests."""
        questions = _make_questions(1, QUESTION_TYPE_MC, 10)
        block_requests = {1: 5}
        tests, messages = generate_all_tests_data(questions, block_requests, 5, self._nop_callback)
        for test in tests:
            assert len(test) == 5
            indices = [q['original_index'] for q in test]
            assert len(indices) == len(set(indices))

    def test_multiple_blocks(self):
        q_block1 = _make_questions(1, QUESTION_TYPE_MC, 12, start_index=0)
        q_block2 = _make_questions(2, QUESTION_TYPE_OE, 12, start_index=12)
        all_questions = q_block1 + q_block2
        block_requests = {1: 3, 2: 2}
        tests, messages = generate_all_tests_data(all_questions, block_requests, 5, self._nop_callback)
        assert len(tests) == 5
        for test in tests:
            assert len(test) == 5

    def test_error_when_k_exceeds_block_size(self):
        questions = _make_questions(1, QUESTION_TYPE_MC, 5)
        block_requests = {1: 10}
        tests, messages = generate_all_tests_data(questions, block_requests, 3, self._nop_callback)
        error_messages = [m for m in messages if m[0] == 'error']
        assert len(error_messages) > 0

    def test_skips_zero_k_blocks(self):
        questions = _make_questions(1, QUESTION_TYPE_MC, 12)
        block_requests = {1: 0}
        tests, messages = generate_all_tests_data(questions, block_requests, 3, self._nop_callback)
        assert len(tests) == 3
        for test in tests:
            assert len(test) == 0

    def test_missing_block_produces_error(self):
        questions = _make_questions(1, QUESTION_TYPE_MC, 12)
        block_requests = {99: 3}
        tests, messages = generate_all_tests_data(questions, block_requests, 2, self._nop_callback)
        error_messages = [m for m in messages if m[0] == 'error']
        assert any("BLOCK_NOT_FOUND_OR_EMPTY" in m[1] for m in error_messages)
