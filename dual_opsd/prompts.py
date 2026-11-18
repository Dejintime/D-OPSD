PROMPT_VERSION = "v1"

# Match the released siyan-zhao/OPSD collator (ae7d2519), reason_first=False.
OPSD_TRANSITION = (
    "\n\nAfter reading the reference solution above, make sure you truly understand "
    "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
    "own words and independent reasoning, derive the same final answer to the problem above. "
    "Think step by step, explore different approaches, and don't be afraid to backtrack "
    "or reconsider if something doesn't work out:\n"
)


def opsd_prompt(tokenizer, problem, solution=None, thinking=False):
    if solution is None:
        content = f"Problem: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    else:
        content = (f"Problem: {problem}\n\nHere is a reference solution to this problem:\n"
                   f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
                   f"{OPSD_TRANSITION}\nPlease reason step by step, and put your final answer within \\boxed{{}}.")
    return tokenizer.apply_chat_template([{"role": "user", "content": content}],
        tokenize=True, add_generation_prompt=True, return_dict=False, enable_thinking=thinking)


def student_prompt(tokenizer, problem, style="legacy", thinking=False):
    if style == 'opsd_eval':
        return tokenizer.apply_chat_template([{'role':'user','content':
            f'{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.'}],
            tokenize=True,add_generation_prompt=True,return_dict=False,enable_thinking=thinking)
    if style == "opsd":
        return opsd_prompt(tokenizer, problem, thinking=thinking)
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": "Solve the math problem step by step. Put the final answer in \\boxed{}."},
         {"role": "user", "content": problem}],
        tokenize=True, add_generation_prompt=True, return_dict=False,
    )


def teacher_prompt(tokenizer, example, privilege="solution_and_answer", style="legacy", thinking=True):
    reference = example["answer"] if privilege == "answer" else example["solution"]
    if style == "opsd":
        return opsd_prompt(tokenizer, example["problem"], reference, thinking)
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": "Solve the math problem step by step. Put the final answer in \\boxed{}. "
          "Reference material is privileged guidance. Continue the student's existing solution prefix; "
          "correct errors when needed rather than copying an incompatible solution."},
         {"role": "user", "content": f"Problem:\n{example['problem']}\n\nReference material:\n{reference}"}],
        tokenize=True, add_generation_prompt=True, return_dict=False,
    )
