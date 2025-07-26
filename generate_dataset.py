"""
Synthetic arithmetic dataset generation with increasing order of difficulties.
Generates problems with varying operations, number of digits, and number of terms.
"""

import random
import json
from typing import List, Dict, Tuple
from datasets import Dataset, DatasetDict
import itertools

SYSTEM_PROMPT = """
Respond ONLY with the answer to the question in the following format:
<answer>
numerical_answer
</answer>
Don't overthink it.
"""

def generate_single_problem(operation: str, num_digits: int, num_terms: int) -> Tuple[str, str]:
    """
    Generate a single arithmetic problem with specified parameters.
    
    Args:
        operation: The operation to use ('+', '-', '*', '/')
        num_digits: Maximum number of digits for each term
        num_terms: Number of terms in the equation
    
    Returns:
        Tuple of (problem_string, answer_string)
    """
    # For multiplication, limit to smaller numbers to avoid extreme values
    if operation == '*':
        # Limit multiplication to 1-2 digits to keep results reasonable
        max_digits = min(num_digits, 2)
        terms = [random.randint(1, 10**max_digits - 1) for _ in range(num_terms)]
    else:
        terms = [random.randint(1, 10**num_digits - 1) for _ in range(num_terms)]
    
    # Build problem string
    problem = f"{terms[0]}"
    for i in range(1, num_terms):
        # Avoid division by zero
        if operation == '/' and terms[i] == 0:
            terms[i] = 1
        problem += f" {operation} {terms[i]}"
    problem += " = "
    
    # Calculate answer
    expression = problem.replace('=', '').strip()
    numeric_answer = eval(expression)
    
    # Format answer string
    if operation == '/':
        if numeric_answer == int(numeric_answer):
            answer_str = str(int(numeric_answer))
        else:
            answer_str = f"{numeric_answer:.2f}".rstrip('0').rstrip('.')
    else:
        answer_str = str(int(numeric_answer))
    
    return problem, answer_str

def generate_problems_for_case(
    operation: str, 
    num_digits: int, 
    num_terms: int, 
    num_problems: int = 50,
    train_split: float = 0.8
) -> Tuple[Dict[str, List], Dict[str, List]]:
    """
    Generate multiple problems for a specific case configuration with stratified split.
    
    Args:
        operation: The operation to use
        num_digits: Maximum number of digits
        num_terms: Number of terms
        num_problems: Number of problems to generate
        train_split: Fraction for training set
    
    Returns:
        Tuple of (train_problems, val_problems) dictionaries
    """
    train_problems = {
        "problem": [],
        "answer": [],
        "operation": [],
        "num_digits": [],
        "num_terms": []
    }
    
    val_problems = {
        "problem": [],
        "answer": [],
        "operation": [],
        "num_digits": [],
        "num_terms": []
    }
    
    # Calculate split sizes
    train_size = int(num_problems * train_split)
    val_size = num_problems - train_size
    
    # Generate training problems
    for _ in range(train_size):
        problem, answer = generate_single_problem(operation, num_digits, num_terms)
        train_problems["problem"].append(problem)
        train_problems["answer"].append(answer)
        train_problems["operation"].append(operation)
        train_problems["num_digits"].append(num_digits)
        train_problems["num_terms"].append(num_terms)
    
    # Generate validation problems
    for _ in range(val_size):
        problem, answer = generate_single_problem(operation, num_digits, num_terms)
        val_problems["problem"].append(problem)
        val_problems["answer"].append(answer)
        val_problems["operation"].append(operation)
        val_problems["num_digits"].append(num_digits)
        val_problems["num_terms"].append(num_terms)
    
    return train_problems, val_problems

def create_prompt_format(problem: str, answer: str) -> Dict[str, any]:
    """
    Format a problem into the required prompt structure.
    
    Args:
        problem: The arithmetic problem string
        answer: The answer string
    
    Returns:
        Dictionary with formatted prompt and answer
    """
    return {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': problem}
        ],
        "answer": answer
    }

def generate_full_dataset(
    operations: List[str] = ['+', '-', '*', '/'],
    digit_ranges: List[int] = [3, 4, 5],
    term_ranges: List[int] = [2, 3, 4],
    problems_per_case: int = 50,
    train_split: float = 0.8,
    val_split: float = 0.2
) -> DatasetDict:
    """
    Generate the complete dataset with all combinations of parameters and stratified splits.
    
    Args:
        operations: List of operations to use
        digit_ranges: List of digit counts to use
        term_ranges: List of term counts to use
        problems_per_case: Number of problems per case
        train_split: Fraction for training set
        val_split: Fraction for validation set
    
    Returns:
        DatasetDict with train and validation splits
    """
    all_train_problems = {
        "problem": [],
        "answer": [],
        "operation": [],
        "num_digits": [],
        "num_terms": []
    }
    
    all_val_problems = {
        "problem": [],
        "answer": [],
        "operation": [],
        "num_digits": [],
        "num_terms": []
    }
    
    # Generate all combinations with stratified splits
    for operation, num_digits, num_terms in itertools.product(operations, digit_ranges, term_ranges):
        train_case_problems, val_case_problems = generate_problems_for_case(
            operation, num_digits, num_terms, problems_per_case, train_split
        )
        
        # Add to master datasets
        for key in all_train_problems.keys():
            all_train_problems[key].extend(train_case_problems[key])
            if val_case_problems[key]:  # Only add if validation problems exist
                all_val_problems[key].extend(val_case_problems[key])
    
    # Create datasets and add prompt formatting
    train_dataset = Dataset.from_dict(all_train_problems)
    train_dataset = train_dataset.map(lambda x: {
        **create_prompt_format(x['problem'], x['answer']),
        'operation': x['operation'],
        'num_digits': x['num_digits'],
        'num_terms': x['num_terms']
    })
    
    # Create validation dataset if there are validation problems
    if any(len(problems) > 0 for problems in all_val_problems.values()):
        val_dataset = Dataset.from_dict(all_val_problems)
        val_dataset = val_dataset.map(lambda x: {
            **create_prompt_format(x['problem'], x['answer']),
            'operation': x['operation'],
            'num_digits': x['num_digits'],
            'num_terms': x['num_terms']
        })
        
        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
    else:
        return DatasetDict({
            'train': train_dataset
        })

def save_dataset(dataset: DatasetDict, output_path: str = "arithmetic_dataset"):
    """
    Save the dataset to disk.
    
    Args:
        dataset: The DatasetDict to save
        output_path: Path where to save the dataset
    """
    dataset.save_to_disk(output_path)
    print(f"Dataset saved to {output_path}")
    print(f"Train size: {len(dataset['train'])}")
    if 'validation' in dataset:
        print(f"Validation size: {len(dataset['validation'])}")
    else:
        print("No validation set created")

def save_dataset_as_jsonl(dataset: DatasetDict, output_path: str = "arithmetic_dataset"):
    """
    Save the dataset as JSONL files for easy inspection.
    
    Args:
        dataset: The DatasetDict to save
        output_path: Base path for the JSONL files
    """
    # Save train set
    train_file = f"{output_path}_train.jsonl"
    with open(train_file, 'w') as f:
        for example in dataset['train']:
            json.dump(example, f)
            f.write('\n')
    print(f"Train set saved to {train_file}")
    
    # Save validation set if it exists
    if 'validation' in dataset:
        val_file = f"{output_path}_validation.jsonl"
        with open(val_file, 'w') as f:
            for example in dataset['validation']:
                json.dump(example, f)
                f.write('\n')
        print(f"Validation set saved to {val_file}")
    
    print(f"JSONL files created with prefix: {output_path}")

if __name__ == "__main__":
    
    digit_ranges = [5, 6, 7, 8]
    term_ranges = [2, 3, 4]
    
    # Generate the complete dataset
    dataset = generate_full_dataset(
        operations=['+'],  # Only addition for now
        digit_ranges=digit_ranges,
        term_ranges=term_ranges,
        problems_per_case=40,
        train_split=0.95,
        val_split=0.05
    )
    
    # Save to disk
    save_dataset(dataset, "arithmetic_dataset")
    
    # Also save as JSONL for easy inspection
    # save_dataset_as_jsonl(dataset, "arithmetic_dataset")
    
    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Total combinations: {len(['+']) * len(digit_ranges) * len(term_ranges)}")
    total_problems = len(dataset['train'])
    if 'validation' in dataset:
        total_problems += len(dataset['validation'])
    print(f"Total problems: {total_problems}")
    
    # Show a few examples
    print("\nSample problems:")
    for i in range(3):
        example = dataset['train'][i]
        print(f"Problem: {example['prompt'][1]['content']}")
        print(f"Answer: {example['answer']}")
        print(f"Operation: {example['operation']}, Digits: {example['num_digits']}, Terms: {example['num_terms']}")
        print()
