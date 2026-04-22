#!/usr/bin/env python3
import os
import re
import json
import openai
import tiktoken
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables from .env file
load_dotenv()

def _make_openai_client(provider_name: str) -> openai.OpenAI:
    """Create an openai.OpenAI client for a known OpenAI-compatible provider."""
    providers = {
        "sambanova":   ("https://api.sambanova.ai/v1",    "SAMBANOVA_API_KEY"),
        "together":    ("https://api.together.xyz/v1",     "TOGETHER_API_KEY"),
        "openai":      ("https://api.openai.com/v1",       "OPENAI_API_KEY"),
        "commonstack": ("https://api.commonstack.ai/v1",   "COMMONSTACK_API_KEY"),
        "openrouter":  ("https://openrouter.ai/api/v1",    "OPENROUTER_API_KEY"),
    }
    if provider_name not in providers:
        raise ValueError(f"Unknown OpenAI-compatible provider: {provider_name}")

    base_url, key_env = providers[provider_name]
    api_key = os.getenv(key_env, "")
    if not api_key:
        raise ValueError(f"{key_env} not found in environment variables")

    kwargs = dict(api_key=api_key, base_url=base_url)
    if provider_name == "openrouter":
        kwargs["timeout"] = openai.Timeout(connect=30.0, read=600, write=600, pool=600)
    return openai.OpenAI(**kwargs)


def _make_internal_openai_client(internal_config: dict) -> openai.OpenAI:
    """Create an openai.OpenAI client for an internal vLLM/TGI endpoint."""
    base_url = internal_config["base_url"]
    api_key_env = internal_config.get("api_key_env")
    api_key = os.getenv(api_key_env, "no-key") if api_key_env else "no-key"
    timeout = internal_config.get("timeout", 120)
    return openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
    )


def _make_client_for_role(provider_name: str, gigachat_config=None, internal_openai_config=None):
    """Create a single client for the given provider name."""
    if provider_name == "gigachat":
        if not gigachat_config:
            raise ValueError("gigachat provider requires 'gigachat' section in config")
        from gigachat_client import make_gigachat_client
        return make_gigachat_client(gigachat_config)
    elif provider_name == "internal_openai":
        if not internal_openai_config:
            raise ValueError("internal_openai provider requires 'internal_openai' section in config")
        return _make_internal_openai_client(internal_openai_config)
    else:
        return _make_openai_client(provider_name)


def initialize_clients(
    api_provider,
    generator_provider=None,
    reflector_provider=None,
    curator_provider=None,
    gigachat_config=None,
    internal_openai_config=None,
):
    """Initialize separate clients for generator, reflector, and curator.

    Per-role provider overrides fall back to api_provider when None.
    """
    gen_prov = generator_provider or api_provider
    ref_prov = reflector_provider or api_provider
    cur_prov = curator_provider or api_provider

    generator_client = _make_client_for_role(gen_prov, gigachat_config, internal_openai_config)
    reflector_client = _make_client_for_role(ref_prov, gigachat_config, internal_openai_config)
    curator_client = _make_client_for_role(cur_prov, gigachat_config, internal_openai_config)

    providers_used = {gen_prov, ref_prov, cur_prov}
    if len(providers_used) == 1:
        print(f"Using {gen_prov} API for all models")
    else:
        print(f"Providers — generator: {gen_prov}, reflector: {ref_prov}, curator: {cur_prov}")
    return generator_client, reflector_client, curator_client

def get_section_slug(section_name):
    """Convert section name to slug format (3-5 chars)"""
    # Common section mappings - updated to match original sections
    slug_map = {
        "financial_strategies_and_insights": "fin",
        "formulas_and_calculations": "calc",
        "code_snippets_and_templates": "code",
        "common_mistakes_to_avoid": "err",
        "problem_solving_heuristics": "prob",
        "context_clues_and_indicators": "ctx",
        "others": "misc",
        "meta_strategies": "meta",
        "introduction": "введ",
        "instructions": "инст",
        "response_format": "фмт",
        "document_catalog": "пд",
        "escalation": "эск",
        "boundaries": "разгр",
        "введение": "введ",
        "инструкции": "инст",
        "формат_ответа": "фмт",
        "перечень_документов": "пд",
        "эскалация": "эск",
        "разграничения": "разгр",
    }
    
    # Clean and convert to snake_case
    clean_name = section_name.lower().strip().replace(" ", "_").replace("&", "and")
    
    if clean_name in slug_map:
        return slug_map[clean_name]
    
    # Generate slug from first letters
    words = clean_name.split("_")
    if len(words) == 1:
        return words[0][:4]
    else:
        return "".join(w[0] for w in words[:5])

def extract_boxed_content(text):
    """Helper function to extract content from \\boxed{} format"""
    pattern = r'\\boxed\{'
    match = re.search(pattern, text)
    if not match:
        return None
    
    start = match.end() - 1  # Position of opening brace
    brace_count = 0
    i = start
    
    while i < len(text):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[start + 1:i]  # Content between braces
        i += 1
    return None

def extract_answer(response):
    """Extract final answer from model response"""
    try:
        # First try JSON parsing
        parsed = json.loads(response)
        answer = str(parsed.get("final_answer", "No final answer found"))
        return answer  
            
    except (json.JSONDecodeError, KeyError, AttributeError):
        # JSON parsing failed, use fallback logic
        matches = re.findall(r"Finish\[(.*?)\]", response)
        if matches:
            answer = matches[-1]
            return answer
        
        # Try to get final answer from JSON style response with regex matching 
        # Try double quotes first
        matches = re.findall(r'"final_answer"\s*:\s*"([^"]*)"', response)
        if matches:
            answer = matches[-1]
            return answer
        
        # Try single quotes
        matches = re.findall(r"'final_answer'\s*:\s*'([^']*)'", response)
        if matches:
            answer = matches[-1]
            return answer
        
        # Handle JSON format without quotes (for simple expressions)
        matches = re.findall(r'[\'"]final_answer[\'"]\s*:\s*([^,}]+)', response)
        if matches:
            answer = matches[-1].strip()
            # Clean up trailing characters
            answer = re.sub(r'[,}]*$', '', answer)
            return answer
        
        # Fallback for "The final answer is: X" pattern with boxed
        final_answer_pattern = r'[Tt]he final answer is:?\s*\$?\\boxed\{'
        match = re.search(final_answer_pattern, response)
        if match:
            # Extract boxed content starting from this match
            remaining_text = response[match.start():]
            boxed_content = extract_boxed_content(remaining_text)
            if boxed_content:
                return boxed_content
        
        # More general pattern for "final answer is X"
        matches = re.findall(r'[Tt]he final answer is:?\s*([^\n.]+)', response)
        if matches:
            answer = matches[-1].strip()
            # Clean up common formatting
            answer = re.sub(r'^\$?\\boxed\{([^}]+)\}\$?$', r'\1', answer)
            answer = answer.replace('$', '').strip()
            if answer:
                return answer
        
        return "No final answer found"
    
enc = tiktoken.get_encoding("cl100k_base")
def count_tokens(prompt: str) -> int:
    return len(enc.encode(prompt))


EVAL_ERROR_SENTINEL = "INCORRECT_DUE_TO_EVALUATION_ERROR"


@dataclass
class EvalSampleOutcome:
    index: int
    final_answer: str
    target: str
    is_correct: bool
    error_message: Optional[str] = None


def evaluate_single_test_sample(args_tuple, data_processor) -> EvalSampleOutcome:
    """
    Evaluate a single test sample - task-agnostic implementation.
    
    Args:
        args_tuple: Tuple of (index, task_dict, generator, playbook, max_tokens, log_dir, use_json_mode)
        data_processor: DataProcessor instance with answer_is_correct method
    """
    (i, task_dict, generator, playbook, max_tokens, log_dir, use_json_mode) = args_tuple
    try:
        context = task_dict["context"]
        question = task_dict["question"]
        target = task_dict["target"]

        gen_response, considered_ids, used_ids, call_info = generator.generate(
            question=question,
            playbook=playbook,
            context=context,
            reflection="(empty)",
            use_json_mode=use_json_mode,
            call_id=f"test_eval_{i}",
            log_dir=log_dir
        )

        final_answer = extract_answer(gen_response)
        is_correct = data_processor.answer_is_correct(final_answer, target)

        return EvalSampleOutcome(
            index=i,
            final_answer=final_answer,
            target=target,
            is_correct=is_correct,
        )

    except Exception as e:
        return EvalSampleOutcome(
            index=i,
            final_answer=EVAL_ERROR_SENTINEL,
            target=task_dict.get("target", ""),
            is_correct=False,
            error_message=f"Error evaluating sample {i}: {type(e).__name__}: {str(e)}",
        )


def evaluate_test_set(data_processor, generator, playbook, test_samples,
                      max_tokens=4096, log_dir=None, max_workers=20, 
                      use_json_mode=False) -> Tuple[Dict, Dict]:
    """
    Parallel evaluation of test set - task-agnostic implementation.
    
    Args:
        data_processor: DataProcessor instance with answer_is_correct and evaluate_accuracy methods
        generator: Generator instance
        playbook: Current playbook string
        test_samples: List of test samples
        max_tokens: Max tokens for generation
        log_dir: Directory for logs
        max_workers: Number of parallel workers
        use_json_mode: Whether to use JSON mode
        
    Returns:
        Tuple of (results_dict, error_logs_dict)
    """
    print(f"\n{'='*40}")
    print(f"EVALUATING TEST SET - {len(test_samples)} samples, {max_workers} workers")
    print(f"{'='*40}")

    args_list = [
        (i, sample, generator, playbook, max_tokens, log_dir, use_json_mode)
        for i, sample in enumerate(test_samples)
    ]

    results = {
        "correct": 0,
        "total": len(test_samples),
        "no_answer": 0,
        "answers": [],
        "targets": [],
        "errors": [],
        "sample_failures": [],
    }

    # Use a wrapper to pass data_processor to the evaluation function
    def eval_wrapper(args_tuple):
        return evaluate_single_test_sample(args_tuple, data_processor)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_args = {
            executor.submit(eval_wrapper, args): args 
            for args in args_list
        }

        for i, future in enumerate(as_completed(future_to_args), 1):
            args_tuple = future_to_args[future]
            try:
                outcome = future.result()
            except Exception as e:
                idx, task_dict, *_ = args_tuple
                outcome = EvalSampleOutcome(
                    index=idx,
                    final_answer=EVAL_ERROR_SENTINEL,
                    target=task_dict.get("target", ""),
                    is_correct=False,
                    error_message=f"Error evaluating sample {idx}: {type(e).__name__}: {str(e)}",
                )

            if outcome.error_message:
                print(outcome.error_message)
                results["sample_failures"].append({
                    "index": outcome.index,
                    "prediction": outcome.final_answer,
                    "ground_truth": outcome.target,
                    "error": outcome.error_message,
                })

            results["correct"] += (1 if outcome.is_correct else 0)
            results["answers"].append(outcome.final_answer)
            results["targets"].append(outcome.target)
            
            if not outcome.is_correct:
                results["errors"].append({
                    "index": outcome.index,
                    "prediction": outcome.final_answer,
                    "ground_truth": outcome.target
                })
            
            if outcome.final_answer == "No final answer found":
                results["no_answer"] += 1

            if i % 50 == 0:
                curr_acc = results["correct"] / results["total"] if results["total"] > 0 else 0
                print(f"Progress: {i}/{len(args_list)}, Accuracy: {curr_acc:.3f}")

    accuracy = (
        data_processor.evaluate_accuracy(results["answers"], results["targets"])
        if results["answers"] and results["targets"]
        else 0.0
    )

    final_results = {
        "accuracy": accuracy,
        "correct": results["correct"],
        "total": results["total"],
        "no_answer": results["no_answer"]
    }

    error_logs = {
        "accuracy": accuracy,
        "errors": results["errors"],
        "sample_failures": results["sample_failures"],
    }

    print(f"\n📊 Final Accuracy: {accuracy:.3f} ({results['correct']}/{results['total']})")

    return final_results, error_logs
