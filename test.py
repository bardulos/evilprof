# test.py
import pandas as pd
import random
import os
import math
from collections import defaultdict

from config import QUESTION_TYPE_MC, QUESTION_TYPE_OE, QUESTION_TYPE_UNKNOWN
from core_logic import generate_all_tests_data

# Constants for test and output file names
TEST_EXCEL_FILE = "test_set_4_by_12_questions.xlsx"
OUTPUT_EXCEL_FILE = "similarity_analysis_unified_dice_mc_15t.xlsx"


def _load_test_questions(status_callback):
    """
    Load questions from TEST_EXCEL_FILE, detect blocks and types.
    Returns (all_questions, blocks_summary, None) or (None, None, error_key).
    Calls status_callback only for critical errors.
    """
    if not os.path.exists(TEST_EXCEL_FILE):
        status_callback("error", "TEST_FILE_NOT_FOUND", filename=TEST_EXCEL_FILE)
        return None, None, "TEST_FILE_NOT_FOUND"
    try:
        _, file_extension = os.path.splitext(TEST_EXCEL_FILE)
        if file_extension.lower() not in ['.xlsx', '.xls']:
            status_callback("error", "FH_UNSUPPORTED_FORMAT", filename=TEST_EXCEL_FILE, extension=file_extension)
            return None, None, "FH_UNSUPPORTED_FORMAT"

        df = pd.read_excel(TEST_EXCEL_FILE, header=None)
        df = df.fillna('').astype(str)

        all_questions = []
        blocks_summary = []
        current_block_id = 1
        current_block_questions = []
        current_block_type = None
        first_question_in_block = True

        df.loc[len(df)] = [""] * df.shape[1]

        for index, row in df.iterrows():
            is_empty_row = all(s is None or str(s).strip() == "" for s in row)
            if is_empty_row:
                if current_block_questions:
                    if current_block_type is None:
                        current_block_type = QUESTION_TYPE_UNKNOWN
                    blocks_summary.append({
                        'block_id': current_block_id,
                        'type': current_block_type,
                        'count': len(current_block_questions)
                    })
                    all_questions.extend(current_block_questions)
                current_block_id += 1
                current_block_questions = []
                current_block_type = None
                first_question_in_block = True
            else:
                row_list = [str(s).strip() for s in row]
                question_text = row_list[0]
                answers = [ans for ans in row_list[1:] if ans]
                if question_text:
                    question_type = QUESTION_TYPE_MC if len(answers) >= 2 else QUESTION_TYPE_OE
                    if first_question_in_block:
                        current_block_type = question_type
                        first_question_in_block = False
                    elif question_type != current_block_type:
                        status_callback("warning", "FH_BLOCK_MIXED_TYPES", block_id=current_block_id, expected=current_block_type, found=question_type, row_num=index + 1)
                        continue
                    question_dict = {
                        'question': question_text,
                        'answers': answers if current_block_type == QUESTION_TYPE_MC else [],
                        'original_index': index,
                        'type': current_block_type,
                        'block_id': current_block_id
                    }
                    current_block_questions.append(question_dict)

        blocks_summary = [b for b in blocks_summary if b['count'] > 0]
        if not all_questions:
            status_callback("error", "FH_NO_VALID_QUESTIONS", filename=TEST_EXCEL_FILE)
            return None, None, "FH_NO_VALID_QUESTIONS"
        return all_questions, blocks_summary, None
    except Exception as e:
        status_callback("error", "TEST_LOAD_ERROR", filename=TEST_EXCEL_FILE, error=str(e))
        return None, None, "TEST_LOAD_ERROR"


def _calculate_dice(set1, set2):
    """Calculate the Sorensen-Dice coefficient between two sets."""
    intersection_cardinality = len(set1.intersection(set2))
    denominator = len(set1) + len(set2)
    if denominator == 0:
        return 1.0
    return 2 * intersection_cardinality / denominator


def _run_single_unified_analysis_for_k(k_per_block, blocks_info, all_questions_list, num_tests_to_generate):
    """
    Run a single similarity analysis for a given k_per_block,
    using the unified logic (WRSwOR or Simple Random) from core_logic.
    Returns dict {distance: avg_dice_index} and list of generation error messages.
    Does NOT call status_callback.
    """
    dice_by_distance = {}
    max_distance_to_check = num_tests_to_generate - 1
    generation_error_messages = []

    def nop_callback(*args, **kwargs):
        pass

    block_requests = {block['block_id']: k_per_block for block in blocks_info if block['count'] >= k_per_block}
    if not block_requests:
        generation_error_messages.append(("error", "STAT_TEST_K_INVALID", {"k": k_per_block}))
        return None, generation_error_messages

    generated_tests_data, gen_messages_internal = generate_all_tests_data(
        all_questions_list, block_requests, num_tests_to_generate, nop_callback
    )
    # Collect warnings (e.g. fallback) and errors
    generation_error_messages.extend([msg for msg in gen_messages_internal if msg[0] in ('error', 'warning')])

    if generated_tests_data is None:
        if not any(m[1] == "STAT_TEST_GENERATION_FAILED_KPB" for m in generation_error_messages):
            generation_error_messages.append(("error", "STAT_TEST_GENERATION_FAILED_KPB", {"k_per_block": k_per_block}))
        return None, generation_error_messages

    test_sets = [set(q['original_index'] for q in test) for test in generated_tests_data]
    for d in range(1, max_distance_to_check + 1):
        dice_indices_for_d = []
        for i in range(num_tests_to_generate - d):
            dice_index = _calculate_dice(test_sets[i], test_sets[i + d])
            dice_indices_for_d.append(dice_index)
        if dice_indices_for_d:
            dice_by_distance[d] = dice_indices_for_d

    avg_dice_results_for_k = {}
    for d, indices in dice_by_distance.items():
        if indices:
            avg = sum(indices) / len(indices)
            avg_dice_results_for_k[d] = avg if not math.isnan(avg) else 0.0
        else:
            avg_dice_results_for_k[d] = None
    return avg_dice_results_for_k, generation_error_messages


# ================================================================
# Monte Carlo Test Orchestrator (run_all_tests)
# ================================================================
def run_all_tests(status_callback, num_monte_carlo_runs=30):
    """
    Orchestrate statistical similarity analysis (Dice) with Monte Carlo,
    varying k_per_block from 1 to 11, using unified logic (WRSwOR/Simple).
    Saves final average results to a formatted Excel file.
    Returns list of tuples (type, key, kwargs_dict) with final summary messages
    and includes a success/failure message for the Excel file.
    Calls status_callback only for critical errors and final messages.
    """
    monte_carlo_summary = []
    results_accumulator = defaultdict(lambda: defaultdict(lambda: {'sum': 0.0, 'count': 0}))
    sampling_method_used = {}
    fallback_counts = defaultdict(int)

    # 1. Load data and block info
    all_questions, blocks_summary, error_key = _load_test_questions(status_callback)
    if error_key:
        monte_carlo_summary.append(("error", "TEST_ABORTED_LOAD_FAILED", {}))
        return monte_carlo_summary, None

    expected_blocks = 4
    expected_q_per_block = 12
    if not blocks_summary or len(blocks_summary) != expected_blocks:
        monte_carlo_summary.append(("error", "TEST_WRONG_BLOCK_COUNT", {"found": len(blocks_summary) if blocks_summary else 0, "expected": expected_blocks}))
        return monte_carlo_summary, None
    for block in blocks_summary:
        if block['count'] != expected_q_per_block:
            monte_carlo_summary.append(("error", "TEST_WRONG_Q_PER_BLOCK_COUNT", {"block_id": block['block_id'], "found": block['count'], "expected": expected_q_per_block}))
            return monte_carlo_summary, None

    # 2. Define parameters
    num_tests_per_sequence = 15
    k_per_block_values = range(1, 12)
    max_distance_overall = 0

    # 3. Outer Monte Carlo loop
    for run in range(1, num_monte_carlo_runs + 1):
        for k_block in k_per_block_values:
            # Determine which method core_logic will use
            method = "WRSwOR" if (k_block * 2 < expected_q_per_block) else "Simple Random"
            if k_block not in sampling_method_used:
                sampling_method_used[k_block] = method

            avg_dice_by_distance, gen_errors = _run_single_unified_analysis_for_k(
                k_block, blocks_summary, all_questions, num_tests_per_sequence
            )
            # Accumulate critical errors from generation
            monte_carlo_summary.extend([msg for msg in gen_errors if msg[0] == 'error'])
            # Count fallback warnings for this k
            if any(m[1] == "BLOCK_FALLBACK_WARNING" for m in gen_errors):
                fallback_counts[k_block] += 1

            if avg_dice_by_distance is not None:
                for d, avg_d in avg_dice_by_distance.items():
                    if avg_d is not None:
                        results_accumulator[k_block][d]['sum'] += avg_d
                        results_accumulator[k_block][d]['count'] += 1
                        max_distance_overall = max(max_distance_overall, d)
            else:
                monte_carlo_summary.append(("warning", "MC_TEST_FAILED_FOR_KPB_IN_RUN", {"k_per_block": k_block, "run": run, "method": method}))

    # 4. Compute final averages and prepare Excel output
    detailed_results_for_excel = []
    sorted_k = sorted(results_accumulator.keys())
    if not max_distance_overall and any(results_accumulator.values()):
        max_distance_overall = 1

    for k_block in sorted_k:
        for d in range(1, max_distance_overall + 1):
            data = results_accumulator[k_block].get(d)
            if data and data['count'] > 0:
                final_avg = data['sum'] / data['count']
                num_samples = data['count']
            else:
                final_avg = None
                num_samples = 0
            detailed_results_for_excel.append({
                'k_per_block': k_block,
                'distance': d,
                'avg_dice': final_avg,
                'num_samples': num_samples,
                'method': sampling_method_used.get(k_block, 'Unknown'),
                'fallback_runs': fallback_counts.get(k_block, 0)
            })

    # 5. Create and save Excel file
    excel_created = False
    excel_filename = None
    if detailed_results_for_excel:
        try:
            df_results = pd.DataFrame(detailed_results_for_excel)
            df_pivot = pd.pivot_table(df_results, values='avg_dice', index='k_per_block', columns='distance')

            method_map = pd.Series(sampling_method_used, name='Metodo Usato')
            fallback_map = pd.Series(fallback_counts, name=f'WRSwOR Fallback Runs (su {num_monte_carlo_runs})')
            df_pivot = df_pivot.join(method_map)
            df_pivot = df_pivot.join(fallback_map)

            # Sort by numeric k_per_block index before formatting
            df_pivot = df_pivot.sort_index(ascending=True)

            # Format index as string after sorting
            df_pivot.index = [f"{k} su {expected_q_per_block}" for k in df_pivot.index]
            df_pivot.index.name = f"k / n (n={expected_q_per_block} per blocco)"

            # Sort and format distance columns
            distance_cols = sorted([col for col in df_pivot.columns if isinstance(col, int)], key=int)
            other_cols = [col for col in df_pivot.columns if not isinstance(col, int)]
            df_pivot = df_pivot.reindex(distance_cols + other_cols, axis=1)
            df_pivot.columns = [f"Distanza {col}" if isinstance(col, int) else col for col in df_pivot.columns]

            df_pivot.to_excel(OUTPUT_EXCEL_FILE, sheet_name='Similarity_Analysis')
            monte_carlo_summary.append(("success", "STAT_TEST_EXCEL_CREATED", {"filename": OUTPUT_EXCEL_FILE}))
            excel_created = True
            excel_filename = OUTPUT_EXCEL_FILE
        except Exception as e:
            monte_carlo_summary.append(("error", "STAT_TEST_EXCEL_SAVE_ERROR", {"filename": OUTPUT_EXCEL_FILE, "error": str(e)}))
    elif any(results_accumulator.values()):
        monte_carlo_summary.append(("warning", "STAT_TEST_NO_DATA_FOR_EXCEL", {}))

    # 6. Final completion message
    monte_carlo_summary.append(("info", "MC_TEST_ALL_COMPLETE", {}))

    return monte_carlo_summary, excel_filename
