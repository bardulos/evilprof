# file_handler.py
import pandas as pd
import os

from config import QUESTION_TYPE_MC, QUESTION_TYPE_OE, QUESTION_TYPE_UNKNOWN


def load_questions_from_excel(uploaded_file, status_callback):
    """
    Load questions/answers from Excel (.xlsx, .xls) or CSV (.csv) file.
    Detects blocks separated by empty rows and determines each block's type.
    Calls status_callback only for significant errors or warnings.
    Returns:
        - all_questions: List of question dicts, each with 'block_id' and 'type'.
        - blocks_summary: List of dicts describing each block {'block_id', 'type', 'count'}.
        - error_key: Error key (str) or None on success.
    """
    if uploaded_file is None:
        return None, None, "UPLOAD_FIRST_WARNING"

    file_name = uploaded_file.name
    df = None

    try:
        _, file_extension = os.path.splitext(file_name)
        file_extension = file_extension.lower()

        if file_extension in ['.xlsx', '.xls']:
            excel_df = pd.read_excel(uploaded_file, header=None)
            df = excel_df.fillna('').astype(str)
        elif file_extension == '.csv':
            try:
                content_bytes = uploaded_file.getvalue()
                content_str = content_bytes.decode('utf-8-sig')
                from io import StringIO
                csv_file_like = StringIO(content_str)
                csv_df = pd.read_csv(csv_file_like, header=None, sep=',', skipinitialspace=True, lineterminator='\n')
                df = csv_df.fillna('').astype(str)
            except Exception as e_comma:
                try:
                    uploaded_file.seek(0)
                    content_bytes = uploaded_file.getvalue()
                    content_str = content_bytes.decode('utf-8-sig')
                    from io import StringIO
                    csv_file_like = StringIO(content_str)
                    csv_df = pd.read_csv(csv_file_like, header=None, sep=';', skipinitialspace=True, lineterminator='\n')
                    df = csv_df.fillna('').astype(str)
                except Exception as e_semicolon:
                    status_callback("error", "FH_CSV_READ_ERROR", filename=file_name, error=f"Comma: {e_comma}, Semicolon: {e_semicolon}")
                    return None, None, "FH_CSV_READ_ERROR"
        else:
            status_callback("error", "FH_UNSUPPORTED_FORMAT", filename=file_name, extension=file_extension)
            return None, None, "FH_UNSUPPORTED_FORMAT"

        # Block analysis logic
        all_questions = []
        blocks_summary = []
        current_block_id = 1
        current_block_questions = []
        current_block_type = None
        first_question_in_block = True

        # Add virtual empty row at the end to finalize last block
        if df is not None:
            df.loc[len(df)] = [""] * df.shape[1]
        else:
            raise ValueError("Failed to read the uploaded file into a DataFrame.")

        for index, row in df.iterrows():
            is_empty_row = all(s is None or str(s).strip() == "" for s in row)

            if is_empty_row:
                # Finalize previous block
                if current_block_questions:
                    if current_block_type is None:
                        current_block_type = QUESTION_TYPE_UNKNOWN
                    blocks_summary.append({
                        'block_id': current_block_id,
                        'type': current_block_type,
                        'count': len(current_block_questions)
                    })
                    all_questions.extend(current_block_questions)
                # Prepare for next block
                current_block_id += 1
                current_block_questions = []
                current_block_type = None
                first_question_in_block = True
            else:
                # Process non-empty row
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

        # Remove empty blocks
        blocks_summary = [b for b in blocks_summary if b['count'] > 0]

        if not all_questions:
            status_callback("error", "FH_NO_VALID_QUESTIONS", filename=file_name)
            return None, None, "FH_NO_VALID_QUESTIONS"

        return all_questions, blocks_summary, None

    except Exception as e:
        status_callback("error", "FH_UNEXPECTED_ERROR", filename=file_name, error=str(e))
        return None, None, "FH_UNEXPECTED_ERROR"
