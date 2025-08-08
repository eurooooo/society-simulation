from qeustionnaire_questions import questionnaire_questions
from typing import List


def parse_binary_strings(binary_strings: List[str], list_len) -> List[List[int]]:
    parsed_lists = []

    for binary_string in binary_strings:
        # Remove any extraneous whitespace or special characters
        cleaned_string = binary_string.replace('\n', '').replace('\t', '').strip()
        try:
            # Split the string into a list of values and convert to integers
            parsed_list = [int(num.strip()) for num in cleaned_string.split(',')]

            # Validate that all elements are binary (0 or 1)
            if all(num in [0, 1] for num in parsed_list):
                parsed_lists.append(parsed_list)
            elif len(parsed_list) != list_len:
                raise ValueError("LLM fails to answer to some questionnaire questions")
            else:
                raise ValueError("Non-binary values detected")

        except Exception as e:
            print(f"Parsing error for: {binary_string}\nError: {e}")
            # If an error occurs, replace with a list of -1s of the same length
            fallback_list = [-1] * list_len
            parsed_lists.append(fallback_list)

    return parsed_lists


def calculate_score(questions, answer_vec: List[int]):
    scores = [0, 0, 0, 0]  # ENFJ
    for text, answer in zip(questions, answer_vec):
        if "[Extroverted]" in text:
            if answer == 1:
                scores[0] += 1
        elif "[Introverted]" in text:
            if answer == 0:
                scores[0] += 1
        elif "[Intuition]" in text:
            if answer == 1:
                scores[1] += 1
        elif "[Sensing]" in text:
            if answer == 0:
                scores[1] += 1
        elif "[Feeling]" in text:
            if answer == 1:
                scores[2] += 1
        elif "[Thinking]" in text:
            if answer == 0:
                scores[2] += 1
        elif "[Judging]" in text:
            if answer == 1:
                scores[3] += 1
        elif "[Perceiving]" in text:
            if answer == 0:
                scores[3] += 1
    return scores


def questionnaire_res_to_latent_score(questions: List[str], responses: List[List[int]]) -> List[List[int]]:
    agents_latent_vec_scores = [calculate_score(questions, response_int) for response_int in responses]
    return agents_latent_vec_scores


if __name__ == '__main__':
    print(questionnaire_res_to_latent_score(questionnaire_questions, llm_output_scores))
