"""
ACE (Agent-Curator-Environment) System
Main orchestrator class for training and testing with playbook-based learning.

This module coordinates three agents:
- Generator: Produces answers using playbook knowledge
- Reflector: Analyzes outputs and tags bullets
- Curator: Updates the playbook based on feedback
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from .core import Generator, Reflector, Curator, BulletpointAnalyzer
from playbook_utils import *
from logger import *
from utils import *


class ACE:
    """
    Main ACE system orchestrator.
    
    Manages the training loop where:
    1. Generator produces answers using playbook
    2. Reflector analyzes answers and tags bullets
    3. Curator updates playbook based on feedback
    
    """
    
    def __init__(
        self,
        api_provider: str,
        generator_model: str,
        reflector_model: str,
        curator_model: str,
        max_tokens: int = 4096,
        initial_playbook: Optional[str] = None,
        use_bulletpoint_analyzer: bool = False,
        bulletpoint_analyzer_threshold: float = 0.90,
        bulletpoint_analyzer_bm25_threshold: float = 0.0,
        bulletpoint_analyzer_block_cross_section: bool = True,
        reflector_reasoning: Optional[dict] = None,
        curator_reasoning: Optional[dict] = None,
        analyzer_reasoning: Optional[dict] = None,
        bulletpoint_analyzer_model: str = 'BAAI/bge-m3',
    ):
        """
        Initialize the ACE system.
        
        Args:
            api_provider: API provider for LLM calls
            generator_model: Model name for generator
            reflector_model: Model name for reflector
            curator_model: Model name for curator
            max_tokens: Maximum tokens for LLM calls
            initial_playbook: Initial playbook content (optional)
            use_bulletpoint_analyzer: Whether to use bulletpoint analyzer for deduplication
            bulletpoint_analyzer_threshold: Similarity threshold for bulletpoint analyzer (0-1)
        """
        # Initialize API clients
        generator_client, reflector_client, curator_client = initialize_clients(api_provider)

        # Initialize the three agents
        self.generator = Generator(generator_client, api_provider, generator_model, max_tokens)
        self.reflector = Reflector(reflector_client, api_provider, reflector_model, max_tokens,
                                   reasoning=reflector_reasoning)
        self.curator = Curator(curator_client, api_provider, curator_model, max_tokens,
                               reasoning=curator_reasoning)
        
        # Initialize bulletpoint analyzer if requested and available
        self.use_bulletpoint_analyzer = use_bulletpoint_analyzer
        self.bulletpoint_analyzer_threshold = bulletpoint_analyzer_threshold
        
        if use_bulletpoint_analyzer:
            self.bulletpoint_analyzer = BulletpointAnalyzer(
                curator_client,
                curator_model,
                max_tokens,
                embedding_model_name=bulletpoint_analyzer_model,
                api_provider=api_provider,
                bm25_threshold=bulletpoint_analyzer_bm25_threshold,
                block_cross_section=bulletpoint_analyzer_block_cross_section,
                reasoning=analyzer_reasoning,
            )
            print(f"✓ BulletpointAnalyzer initialized (threshold={bulletpoint_analyzer_threshold})")
        else:
            self.bulletpoint_analyzer = None
        
        # Store configuration
        self.generator_client = generator_client
        self.reflector_client = reflector_client
        self.curator_client = curator_client
        self.max_tokens = max_tokens
        
        # Initialize playbook
        if initial_playbook:
            self.playbook = initial_playbook
        else:
            self.playbook = self._initialize_empty_playbook()
        
        self.best_playbook = self.playbook
        # Track global bullet ID — rescue from loaded playbook so resumed runs
        # don't reuse IDs that already exist (would collide with curator output).
        self.next_global_id = get_next_global_id(self.playbook)

    def _initialize_empty_playbook(self) -> str:
        """Initialize an empty playbook with standard sections."""
        return """## STRATEGIES & INSIGHTS

## FORMULAS & CALCULATIONS

## CODE SNIPPETS & TEMPLATES

## COMMON MISTAKES TO AVOID

## PROBLEM-SOLVING HEURISTICS

## CONTEXT CLUES & INDICATORS

## OTHERS"""
    
    def _extract_config_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract common configuration parameters.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dictionary with extracted parameters
        """
        return {
            'num_epochs': config.get('num_epochs', 1),
            'max_num_rounds': config.get('max_num_rounds', 3),
            'curator_frequency': config.get('curator_frequency', 1),
            'eval_steps': config.get('eval_steps', 100),
            'save_steps': config.get('save_steps', 50),
            'token_budget': config.get('playbook_token_budget', 80000),
            'task_name': config.get('task_name', 'default'),
            'use_json_mode': config.get('json_mode', False),
            'no_ground_truth': config.get('no_ground_truth', False),
            'save_dir': config.get('save_dir', './results'),
            'test_workers': config.get('test_workers', 20),
            'use_bulletpoint_analyzer': config.get('use_bulletpoint_analyzer', False),
            'bulletpoint_analyzer_threshold': config.get('bulletpoint_analyzer_threshold', 0.90),
            'bulletpoint_analyzer_bm25_threshold': config.get('bulletpoint_analyzer_bm25_threshold', 0.0),
            'batch_size': config.get('batch_size', 1),
            'skip_first_train_samples': config.get('skip_first_train_samples', 0),
            'resume_epoch': config.get('resume_epoch', 1),
            'prune_max_active_bullets_per_section': config.get('prune_max_active_bullets_per_section', 40),
            'prune_warmup_window': config.get('prune_warmup_window', 50),
            'prune_min_observations': config.get('prune_min_observations', 3)
        }
    
    def _setup_paths(self, save_dir: str, task_name: str, mode: str) -> Tuple[str, str]:
        """
        Setup logging paths and directories.
        
        Args:
            save_dir: Base path for saving results
            task_name: task name
            mode: 'offline', 'online', or 'eval_only'
            
        Returns:
            Tuple of (usage_log_path, playbook_dir)
        """
        # Create timestamped run folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = f"ace_run_{timestamp}_{task_name}_{mode}"
        save_path = os.path.join(save_dir, run_folder)
        os.makedirs(save_path, exist_ok=True)
        log_dir = os.path.join(save_path, "detailed_llm_logs")
        os.makedirs(log_dir, exist_ok=True)

        if mode == "eval_only":
            return save_path, log_dir

        usage_log_path = os.path.join(save_path, "bullet_usage_log.jsonl")
        playbook_dir = os.path.join(save_path, "intermediate_playbooks")
        os.makedirs(playbook_dir, exist_ok=True)
        
        return save_path, usage_log_path, playbook_dir, log_dir

    def _build_test_cache_key(
        self,
        playbook: str,
        test_samples: List[Dict[str, Any]],
        use_json_mode: bool,
    ) -> tuple[str, str]:
        """Fingerprint the exact generator inputs used during evaluation."""
        rendered_playbook = render_minimal_playbook(playbook)
        samples_payload = json.dumps(
            test_samples,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        cache_key_str = "\0".join([
            rendered_playbook,
            self.generator.model,
            str(use_json_mode),
            samples_payload,
        ])
        cache_key = hashlib.sha256(cache_key_str.encode("utf-8")).hexdigest()[:16]
        return cache_key, rendered_playbook

    def _build_offline_training_results(
        self,
        best_accuracy: float,
        *,
        results: Optional[List[Dict[str, Any]]] = None,
        pre_train_post_train_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Return the canonical offline-training payload with a compat alias."""
        payload: Dict[str, Any] = {
            "best_validation_accuracy": best_accuracy,
            "best_accuracy": best_accuracy,
        }
        if results is not None:
            payload["results"] = results
        if pre_train_post_train_results is not None:
            payload["pre_train_post_train_results"] = pre_train_post_train_results
        return payload

    def _write_offline_progress(
        self,
        save_path: str,
        best_accuracy: float,
        results: List[Dict[str, Any]],
        error_logs: List[Dict[str, Any]],
    ) -> None:
        """Persist offline training metrics in the canonical schema."""
        with open(os.path.join(save_path, "train_results.json"), "w", encoding="utf-8") as f:
            json.dump(
                self._build_offline_training_results(best_accuracy, results=results),
                f,
                indent=2,
                ensure_ascii=False,
            )
        with open(os.path.join(save_path, "val_results.json"), "w", encoding="utf-8") as f:
            json.dump(error_logs, f, indent=2, ensure_ascii=False)
    
    def run(
        self,
        mode: str,
        train_samples: Optional[List[Dict[str, Any]]] = None,
        val_samples: Optional[List[Dict[str, Any]]] = None,
        test_samples: Optional[List[Dict[str, Any]]] = None,
        data_processor = None,
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Main entrypoint for running ACE system in different modes.
        
        Args:
            mode: Run mode - 'offline', 'online', or 'eval_only'
            train_samples: Training samples (required for offline mode)
            val_samples: Validation samples (required for offline mode)
            test_samples: Test samples (required for online and eval_only modes)
            data_processor: Data processor instance for the task
            config: Configuration dictionary
            
        Returns:
            Dictionary with results depending on the mode
        """
        # Validate inputs
        if mode not in ['offline', 'online', 'eval_only']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'offline', 'online', or 'eval_only'")
        
        if mode == 'offline' and (train_samples is None or val_samples is None):
            raise ValueError("Offline mode requires train_samples and val_samples")
        
        if mode == 'online' and test_samples is None:
            raise ValueError("Online mode requires test_samples")
        
        if mode == 'eval_only' and test_samples is None:
            raise ValueError("eval_only mode requires test_samples")
        
        # Extract configuration
        config_params = self._extract_config_params(config)
        task_name = config_params['task_name']
        save_dir = config_params['save_dir']
        
        # Setup paths based on mode
        if mode == 'eval_only':
            save_path, log_dir = self._setup_paths(save_dir, task_name, mode)
            usage_log_path = None
            playbook_dir = None
        else:
            save_path, usage_log_path, playbook_dir, log_dir = self._setup_paths(save_dir, task_name, mode)
        
        # Save configuration
        config_path = os.path.join(save_path, "run_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({
                "task_name": task_name,
                "mode": mode,
                "generator_model": self.generator.model,
                "reflector_model": self.reflector.model,
                "curator_model": self.curator.model,
                "config": config,
            }, f, indent=2, ensure_ascii=False)
        
        # Print initial banner
        print(f"\n{'='*60}")
        print(f"ACE SYSTEM - {mode.upper().replace('_', ' ')} MODE")
        print(f"{'='*60}")
        print(f"Task: {task_name}")
        if mode == 'offline':
            print(f"Train samples: {len(train_samples)}")
            print(f"Validation samples: {len(val_samples)}")
            if test_samples:
                print(f"Test samples: {len(test_samples)}")
        elif mode == 'online':
            print(f"Test samples (used for training and testing): {len(test_samples)}")
        else:  # eval_only
            print(f"Test samples: {len(test_samples)}")
        print(f"{'='*60}\n")
        
        # Execute based on mode
        results = {}
        
        if mode == 'offline':
            # OFFLINE MODE WORKFLOW
            # 1. Run initial test if test_samples provided
            if test_samples:
                print(f"\n{'='*60}")
                print(f"INITIAL TEST (before training)")
                print(f"{'='*60}\n")
                initial_test_results = self._run_test(
                    test_samples=test_samples,
                    data_processor=data_processor,
                    playbook=self.playbook,
                    config=config,
                    log_dir=log_dir,
                    save_path=save_path,
                    prefix="initial"
                )
                results['initial_test_results'] = initial_test_results
                print(f"Initial Test Accuracy: {initial_test_results['accuracy']:.3f}\n")
            
            # 2. Run offline training
            print(f"\n{'='*60}")
            print(f"STARTING OFFLINE TRAINING")
            print(f"{'='*60}\n")
            training_results = self._offline_train(
                train_samples=train_samples,
                val_samples=val_samples,
                data_processor=data_processor,
                config=config,
                save_path=save_path,
                usage_log_path=usage_log_path,
                playbook_dir=playbook_dir,
                log_dir=log_dir
            )
            results['training_results'] = training_results
            
            # 3. Run final test if test_samples provided
            if test_samples:
                print(f"\n{'='*60}")
                print(f"FINAL TEST (with best playbook)")
                print(f"{'='*60}\n")
                final_test_results = self._run_test(
                    test_samples=test_samples,
                    data_processor=data_processor,
                    playbook=self.best_playbook,
                    config=config,
                    log_dir=log_dir,
                    save_path=save_path,
                    prefix="final"
                )
                results['final_test_results'] = final_test_results
                print(f"Final Test Accuracy: {final_test_results['accuracy']:.3f}\n")
        
        elif mode == 'online':
            # ONLINE MODE WORKFLOW
            # 1. Run initial test
            print(f"\n{'='*60}")
            print(f"INITIAL TEST (before training)")
            print(f"{'='*60}\n")
            initial_test_results = self._run_test(
                test_samples=test_samples,
                data_processor=data_processor,
                playbook=self.playbook,
                config=config,
                log_dir=log_dir,
                save_path=save_path,
                prefix="initial"
            )
            results['initial_test_results'] = initial_test_results
            print(f"Initial Test Accuracy: {initial_test_results['accuracy']:.3f}\n")
            
            # 2. Run online training and testing
            print(f"\n{'='*60}")
            print(f"STARTING ONLINE TRAIN AND TEST")
            print(f"{'='*60}\n")
            online_results = self._online_train_and_test(
                test_samples=test_samples,
                data_processor=data_processor,
                config=config,
                save_path=save_path,
                usage_log_path=usage_log_path,
                playbook_dir=playbook_dir,
                log_dir=log_dir
            )
            results['online_test_results'] = online_results
        
        else:  # eval_only
            # EVAL ONLY MODE WORKFLOW
            print(f"\n{'='*60}")
            print(f"RUNNING TEST")
            print(f"{'='*60}\n")
            test_results = self._run_test(
                test_samples=test_samples,
                data_processor=data_processor,
                playbook=self.playbook,
                config=config,
                log_dir=log_dir,
                save_path=save_path,
                prefix="test"
            )
            results['test_results'] = test_results
        
        # Save consolidated results
        final_results_path = os.path.join(save_path, "final_results.json")
        with open(final_results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Print final summary
        print(f"\n{'='*60}")
        print(f"RUN COMPLETE")
        print(f"{'='*60}")
        print(f"Mode: {mode.upper().replace('_', ' ')}")
        if mode == 'offline':
            best_val_accuracy = results['training_results'].get(
                'best_validation_accuracy',
                results['training_results'].get('best_accuracy', 0.0),
            )
            print(f"Best Validation Accuracy: {best_val_accuracy:.3f}")
            if test_samples:
                print(f"Initial Test Accuracy: {results['initial_test_results']['accuracy']:.3f}")
                print(f"Final Test Accuracy: {results['final_test_results']['accuracy']:.3f}")
        elif mode == 'online':
            print(f"Initial Test Accuracy: {results['initial_test_results']['accuracy']:.3f}")
            print(f"Final Test Accuracy: {results['online_test_results']['accuracy']:.3f}")
        else:  # eval_only
            print(f"Test Accuracy: {results['test_results']['accuracy']:.3f}")
        print(f"Results saved to: {save_path}")
        print(f"{'='*60}\n")
        
        return results
    
    def _run_test(
        self,
        test_samples: List[Dict[str, Any]],
        data_processor,
        playbook: str,
        config: Dict[str, Any],
        log_dir: str,
        save_path: str,
        prefix: str = "test"
    ) -> Dict[str, Any]:
        """
        Run testing
        
        Args:
            test_samples: List of test samples
            data_processor: Data processor instance for the task
            playbook: Playbook to use for testing
            config: Configuration dictionary
            log_dir: Directory for detailed logs
            save_path: Path to save results
            prefix: Prefix for saved files (e.g., 'initial', 'final', 'test')
            
        Returns:
            Dictionary with test results
        """
        config_params = self._extract_config_params(config)
        use_json_mode = config_params['use_json_mode']
        test_workers = config_params['test_workers']

        # Test cache is keyed by the exact generator inputs, not just set size.
        save_dir_root = config_params.get('save_dir', os.path.dirname(save_path))
        cache_dir = os.path.join(save_dir_root, "_test_cache")
        cache_key, rendered_playbook = self._build_test_cache_key(
            playbook,
            test_samples,
            use_json_mode,
        )
        cache_file = os.path.join(cache_dir, f"{prefix}_{cache_key}.json")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                print(f"📦 Test cache HIT ({prefix}): {cache_file}")
                print(f"   accuracy={cached.get('accuracy', 0):.3f}  correct={cached.get('correct')}/{cached.get('total')}")
                # Also write per-run copy for audit (mirrors normal save logic)
                test_results_path = os.path.join(save_path, f"{prefix}_test_results.json")
                with open(test_results_path, "w", encoding="utf-8") as f:
                    json.dump({"test_results": cached, "error_log": {}, "from_cache": cache_file}, f, indent=2, ensure_ascii=False)
                return cached
            except Exception as e:
                print(f"⚠️  Cache read failed ({e}), re-running test")

        test_results, test_error_log = evaluate_test_set(
            data_processor,
            self.generator,
            rendered_playbook,
            test_samples,
            self.max_tokens,
            log_dir,
            max_workers=test_workers,
            use_json_mode=use_json_mode
        )

        # Save test results
        test_results_path = os.path.join(save_path, f"{prefix}_test_results.json")
        with open(test_results_path, "w", encoding="utf-8") as f:
            json.dump({
                "test_results": test_results,
                "error_log": test_error_log,
            }, f, indent=2, ensure_ascii=False)

        # Write to shared cache for future runs with the same (playbook, model, set size)
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(test_results, f, indent=2, ensure_ascii=False)
            print(f"📦 Test cache saved: {cache_file}")
        except Exception as e:
            print(f"⚠️  Cache write failed: {e}")

        return test_results
    
    def _train_single_sample(
        self,
        task_dict: Dict[str, Any],
        data_processor,
        step_id: str,
        epoch: int,
        step: int,
        usage_log_path: str,
        log_dir: str,
        config_params: Dict[str, Any],
        total_samples: int
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Train on a single sample with reflection and curation.
        
        Args:
            task_dict: Sample dictionary with question, context, target
            data_processor: Data processor for evaluation
            step_id: Identifier string for this step (e.g., "train_e_1_s_10" or "online_train_w_1_s_5")
            epoch: Current epoch number
            step: Current step number
            usage_log_path: Path for bullet usage logging
            log_dir: Path for logging directory
            config_params: Configuration parameters dictionary
            total_samples: Total number of samples in dataset
            
        Returns:
            Tuple of (pre_train_answer, post_train_answer, tracking_dict)
        """
        # Extract configuration
        max_num_rounds = config_params['max_num_rounds']
        curator_frequency = config_params['curator_frequency']
        token_budget = config_params['token_budget']
        use_json_mode = config_params['use_json_mode']
        no_ground_truth = config_params['no_ground_truth']
        
        # Extract sample data
        question = task_dict.get("question", "")
        context = task_dict.get("context", "")
        target = task_dict.get("target", "")
        
        # STEP 1: Initial generation (pre-train)
        print("Generating initial answer...")
        gen_response, considered_ids, used_ids, call_info = self.generator.generate(
            question=question,
            playbook=render_minimal_playbook(self.playbook),
            context=context,
            reflection="(empty)",
            use_json_mode=use_json_mode,
            call_id=f"{step_id}_gen_initial",
            log_dir=log_dir
        )

        # Extract answer and check correctness
        final_answer = extract_answer(gen_response)
        is_correct = data_processor.answer_is_correct(final_answer, target)
        pre_train_answer = final_answer

        print(f"Correct: {is_correct}")

        # Log bullet usage
        log_bullet_usage(usage_log_path, epoch, step, task_dict, considered_ids,
                       playbook=self.playbook, is_correct=is_correct)
        
        # Track pre-train result
        tracking_dict = {
            "pre_train_result": {
                "final_answer": final_answer,
                "is_correct": is_correct,
                "playbook_num_tokens": count_tokens(self.playbook),
                "playbook_length": len(self.playbook)
            }
        }
        
        reflection_content = "(empty)"
        
        # STEP 2: Reflection and regeneration
        if not is_correct:
            # For incorrect answers - iterate reflection rounds
            for round_num in range(max_num_rounds):
                print(f"Reflection round {round_num + 1}/{max_num_rounds}")

                # Get bullets for reflector (considered set — wider than just used)
                playbook_bullets = extract_playbook_bullets(
                    self.playbook, considered_ids
                )

                # Reflect on error
                reflection_content, bullet_tags, _ = self.reflector.reflect(
                    question=question,
                    reasoning_trace=gen_response,
                    predicted_answer=final_answer,
                    ground_truth=target if not no_ground_truth else None,
                    environment_feedback="Predicted answer does not match ground truth",
                    bullets_considered=playbook_bullets,
                    use_ground_truth=not no_ground_truth,
                    use_json_mode=use_json_mode,
                    call_id=f"{step_id}_round_{round_num}",
                    log_dir=log_dir
                )

                # Update bullet counts + recency
                if bullet_tags or considered_ids:
                    self.playbook = update_bullet_counts(
                        self.playbook, bullet_tags,
                        considered_bullet_ids=considered_ids,
                        used_bullet_ids=used_ids,
                        current_step=step,
                    )

                # Regenerate with reflection
                gen_response, considered_ids, used_ids, _ = self.generator.generate(
                    question=question,
                    playbook=render_minimal_playbook(self.playbook),
                    context=context,
                    reflection=reflection_content,
                    use_json_mode=use_json_mode,
                    call_id=f"{step_id}_post_reflect_round_{round_num}",
                    log_dir=log_dir
                )

                final_answer = extract_answer(gen_response)

                if data_processor.answer_is_correct(final_answer, target):
                    print(f"Corrected after reflection round {round_num + 1}!")
                    is_correct = True
                    break

        else:
            # For correct answers - still run reflector to tag helpful bullets
            playbook_bullets = extract_playbook_bullets(
                self.playbook, considered_ids
            )

            reflection_content, bullet_tags, _ = self.reflector.reflect(
                question=question,
                reasoning_trace=gen_response,
                predicted_answer=final_answer,
                ground_truth=target if not no_ground_truth else None,
                environment_feedback="Predicted answer matches ground truth",
                bullets_considered=playbook_bullets,
                use_ground_truth=not no_ground_truth,
                use_json_mode=use_json_mode,
                call_id=f"{step_id}_reflect_on_correct",
                log_dir=log_dir
            )

            # Update bullet counts + recency
            if bullet_tags or considered_ids:
                self.playbook = update_bullet_counts(
                    self.playbook, bullet_tags,
                    considered_bullet_ids=considered_ids,
                    used_bullet_ids=used_ids,
                    current_step=step,
                )

            # Log with reflection
            log_bullet_usage(usage_log_path, epoch, step, task_dict, considered_ids,
                           playbook=self.playbook,
                           reflection_content=reflection_content,
                           is_correct=is_correct)
        
        # STEP 3: Curator - Periodically update playbook
        if step % curator_frequency == 0:
            print(f"\n--- Running Curator at step {step} ---")
            
            stats = get_playbook_stats(self.playbook)
            
            self.playbook, self.next_global_id, operations, _ = self.curator.curate(
                current_playbook=self.playbook,
                recent_reflection=reflection_content,
                question_context=context,
                current_step=step,
                total_samples=total_samples,
                token_budget=token_budget,
                playbook_stats=stats,
                use_ground_truth=not no_ground_truth,
                use_json_mode=use_json_mode,
                call_id=step_id,
                log_dir=log_dir,
                next_global_id=self.next_global_id
            )
            
            # Run bulletpoint analyzer if enabled
            if self.use_bulletpoint_analyzer and self.bulletpoint_analyzer:
                print(f"  Running BulletpointAnalyzer (threshold={self.bulletpoint_analyzer_threshold})...")
                self.playbook = self.bulletpoint_analyzer.analyze(
                    playbook=self.playbook,
                    threshold=self.bulletpoint_analyzer_threshold,
                    merge=True,
                    log_dir=log_dir,
                    call_id_prefix=f"analyzer_{step_id}",
                )

            # PRUNE: deterministic lifecycle archive (fork §3.2 grow-and-refine)
            self.playbook, archived = prune_playbook(
                self.playbook, current_step=step,
                max_active_bullets_per_section=config_params.get('prune_max_active_bullets_per_section', 40),
                warmup_window=config_params.get('prune_warmup_window', 50),
                min_observations=config_params.get('prune_min_observations', 3),
            )
            if archived:
                preview = archived[:5]
                more = "…" if len(archived) > 5 else ""
                print(f"  Archived {len(archived)} bullets: {preview}{more}")

        # STEP 4: Post-curator generation
        gen_response, _, _, _ = self.generator.generate(
            question=question,
            playbook=render_minimal_playbook(self.playbook),
            context=context,
            reflection="(empty)",
            use_json_mode=use_json_mode,
            call_id=f"{step_id}_post_curate",
            log_dir=log_dir
        )
        
        final_answer = extract_answer(gen_response)
        post_train_answer = final_answer
        
        post_train_is_correct = data_processor.answer_is_correct(final_answer, target)
        tracking_dict["post_train_result"] = {
            "final_answer": final_answer,
            "is_correct": post_train_is_correct,
            "playbook_num_tokens": count_tokens(self.playbook),
            "playbook_length": len(self.playbook)
        }
        
        return pre_train_answer, post_train_answer, tracking_dict

    def _train_sample_phase1(
        self,
        task_dict: Dict[str, Any],
        snapshot_playbook: str,
        data_processor,
        step_id: str,
        log_dir: str,
        config_params: Dict[str, Any],
        step: int = 0,
    ) -> Dict[str, Any]:
        """Phase 1 of batched training: Generator + Reflection on a *snapshot*
        of the playbook. Does NOT mutate self.playbook (safe to run in
        parallel across samples).

        Returned dict carries enough state for the sequential phase 2
        (curator) and parallel phase 3 (post-curate generation).
        """
        max_num_rounds = config_params['max_num_rounds']
        use_json_mode = config_params['use_json_mode']
        no_ground_truth = config_params['no_ground_truth']

        question = task_dict.get("question", "")
        context = task_dict.get("context", "")
        target = task_dict.get("target", "")

        local_playbook = snapshot_playbook
        round_evidence: List[Dict[str, Any]] = []
        # Collect considered/used across reflection rounds; last state is what we publish
        last_considered_ids: List[str] = []
        last_used_ids: List[str] = []
        reflection_content = "(empty)"

        gen_response, considered_ids, used_ids, _ = self.generator.generate(
            question=question, playbook=render_minimal_playbook(local_playbook), context=context,
            reflection="(empty)", use_json_mode=use_json_mode,
            call_id=f"{step_id}_gen_initial", log_dir=log_dir,
        )
        last_considered_ids, last_used_ids = considered_ids, used_ids
        log_considered_ids = list(considered_ids)
        pre_train_answer = extract_answer(gen_response)
        is_correct = data_processor.answer_is_correct(pre_train_answer, target)
        final_answer = pre_train_answer

        if not is_correct:
            for round_num in range(max_num_rounds):
                playbook_bullets = extract_playbook_bullets(local_playbook, considered_ids)
                reflection_content, bullet_tags, _ = self.reflector.reflect(
                    question=question, reasoning_trace=gen_response,
                    predicted_answer=final_answer,
                    ground_truth=target if not no_ground_truth else None,
                    environment_feedback="Predicted answer does not match ground truth",
                    bullets_considered=playbook_bullets,
                    use_ground_truth=not no_ground_truth,
                    use_json_mode=use_json_mode,
                    call_id=f"{step_id}_round_{round_num}", log_dir=log_dir,
                )
                if bullet_tags or considered_ids:
                    round_evidence.append({
                        "bullet_tags": list(bullet_tags or []),
                        "considered_ids": list(considered_ids),
                        "used_ids": list(used_ids),
                    })
                    local_playbook = update_bullet_counts(
                        local_playbook, bullet_tags,
                        considered_bullet_ids=considered_ids,
                        used_bullet_ids=used_ids,
                        current_step=step,
                    )

                gen_response, considered_ids, used_ids, _ = self.generator.generate(
                    question=question, playbook=render_minimal_playbook(local_playbook), context=context,
                    reflection=reflection_content, use_json_mode=use_json_mode,
                    call_id=f"{step_id}_post_reflect_round_{round_num}", log_dir=log_dir,
                )
                last_considered_ids, last_used_ids = considered_ids, used_ids
                final_answer = extract_answer(gen_response)
                if data_processor.answer_is_correct(final_answer, target):
                    is_correct = True
                    break
        else:
            playbook_bullets = extract_playbook_bullets(local_playbook, considered_ids)
            reflection_content, bullet_tags, _ = self.reflector.reflect(
                question=question, reasoning_trace=gen_response,
                predicted_answer=final_answer,
                ground_truth=target if not no_ground_truth else None,
                environment_feedback="Predicted answer matches ground truth",
                bullets_considered=playbook_bullets,
                use_ground_truth=not no_ground_truth,
                use_json_mode=use_json_mode,
                call_id=f"{step_id}_reflect_on_correct", log_dir=log_dir,
            )
            if bullet_tags or considered_ids:
                round_evidence.append({
                    "bullet_tags": list(bullet_tags or []),
                    "considered_ids": list(considered_ids),
                    "used_ids": list(used_ids),
                })

        return {
            'pre_train_answer': pre_train_answer,
            'is_correct_pre': data_processor.answer_is_correct(pre_train_answer, target),
            'reflection_content': reflection_content,
            'round_evidence': round_evidence,
            'gen_response': gen_response,
            'considered_ids': last_considered_ids,
            'used_ids': last_used_ids,
            'log_considered_ids': log_considered_ids,
        }

    def _train_sample_phase3(
        self,
        task_dict: Dict[str, Any],
        current_playbook: str,
        step_id: str,
        log_dir: str,
        use_json_mode: bool,
    ) -> str:
        """Phase 3: post-curate generation against the live playbook.
        Read-only on self.playbook (passed in as `current_playbook`)."""
        question = task_dict.get("question", "")
        context = task_dict.get("context", "")
        gen_response, _, _, _ = self.generator.generate(
            question=question, playbook=render_minimal_playbook(current_playbook), context=context,
            reflection="(empty)", use_json_mode=use_json_mode,
            call_id=f"{step_id}_post_curate", log_dir=log_dir,
        )
        return extract_answer(gen_response)

    def _offline_train_batched(
        self,
        train_samples: List[Dict[str, Any]],
        val_samples: List[Dict[str, Any]],
        data_processor,
        config: Dict[str, Any],
        save_path: str,
        usage_log_path: str,
        playbook_dir: str,
        log_dir: str,
        batch_size: int,
    ) -> Dict[str, Any]:
        """Batched offline training.

        Per batch of K samples:
          phase 1 (parallel): Generator + Reflector on a frozen playbook snapshot
          phase 2 (sequential): apply collected bullet_tags + Curator
          phase 3 (parallel): post-curate Generator on the new playbook

        At batch_size=1 the result is byte-identical to the sequential path
        in `_train_single_sample` (modulo phase 2 ordering, which is identical
        for K=1).
        """
        config_params = self._extract_config_params(config)
        task_name = config_params['task_name']
        num_epochs = config_params['num_epochs']
        eval_steps = config_params['eval_steps']
        save_steps = config_params['save_steps']
        test_workers = config_params['test_workers']
        use_json_mode = config_params['use_json_mode']
        curator_frequency = config_params['curator_frequency']
        no_ground_truth = config_params['no_ground_truth']
        token_budget = config_params['token_budget']
        skip_n = config_params.get('skip_first_train_samples', 0)
        start_epoch = config_params.get('resume_epoch', 1)

        results: List[Dict[str, Any]] = []
        pre_train_post_train_results: List[Dict[str, Any]] = []
        error_logs: List[Dict[str, Any]] = []
        best_accuracy = 0.0
        self.best_playbook = self.playbook

        original_total = len(train_samples)
        if skip_n > 0:
            print(f"Resume: skipping first {skip_n} train samples")
            train_samples = train_samples[skip_n:]

        print(f"Epochs: {start_epoch}..{start_epoch + num_epochs - 1} (run will do {num_epochs} epochs)")
        print(f"Train samples per epoch: {len(train_samples)} (of original {original_total})")
        print(f"Val samples: {len(val_samples)}")
        print(f"Batch size (parallel phase 1+3): {batch_size}")
        print(f"Curator frequency: every {curator_frequency} steps")
        print(f"Evaluation frequency: every {eval_steps} steps\n")

        for epoch in range(start_epoch, start_epoch + num_epochs):
            print(f"\n{'='*60}\nEPOCH {epoch} (of {start_epoch + num_epochs - 1})\n{'='*60}")

            epoch_answers_pre_train: List[str] = []
            epoch_targets_pre_train: List[str] = []
            epoch_answers_post_train: List[str] = []
            epoch_targets_post_train: List[str] = []

            # Monotonic step counter across epochs: add (epoch-1) * original_total
            # so bullets' last_used_step never regresses when the inner loop wraps.
            epoch_offset = (epoch - 1) * original_total
            for batch_start in range(0, len(train_samples), batch_size):
                batch = train_samples[batch_start:batch_start + batch_size]
                batch_steps = list(range(
                    batch_start + 1 + skip_n + epoch_offset,
                    batch_start + 1 + skip_n + epoch_offset + len(batch),
                ))
                print(f"\n--- Batch steps {batch_steps[0]}-{batch_steps[-1]} (epoch {epoch}, offset={epoch_offset}) ---")

                snapshot = self.playbook

                # PHASE 1: parallel gen + reflect on snapshot
                with ThreadPoolExecutor(max_workers=len(batch)) as exe:
                    futs = {
                        exe.submit(
                            self._train_sample_phase1,
                            sample, snapshot, data_processor,
                            f"train_e_{epoch}_s_{step}", log_dir, config_params, step,
                        ): step
                        for sample, step in zip(batch, batch_steps)
                    }
                    phase1_by_step = {futs[f]: f.result() for f in as_completed(futs)}

                # PHASE 2a: sequential bullet_tag merge + recency update (no LLM)
                for sample, step in zip(batch, batch_steps):
                    p1 = phase1_by_step[step]
                    # Replay each reflection round against the live playbook using
                    # the exact evidence collected in that round.
                    for evidence in p1['round_evidence']:
                        self.playbook = update_bullet_counts(
                            self.playbook, evidence['bullet_tags'],
                            considered_bullet_ids=evidence['considered_ids'],
                            used_bullet_ids=evidence['used_ids'],
                            current_step=step,
                        )
                    log_bullet_usage(
                        usage_log_path, epoch, step, sample, p1['log_considered_ids'],
                        playbook=self.playbook,
                        reflection_content=p1['reflection_content'],
                        is_correct=p1['is_correct_pre'],
                    )

                # PHASE 2b: SINGLE curator call per batch with aggregated reflections.
                # Previously we ran 10 curators in parallel on the same playbook snapshot
                # but each with one sample's reflection — measured 48% duplicate UPDATEs
                # (same bullet hit 2-4x per batch because curators couldn't coordinate).
                # Aggregating reflections gives curator the full batch context and removes
                # redundancy. Triggers every batch whose last step is divisible by
                # curator_frequency (default 1 = every batch).
                if batch_steps[-1] % curator_frequency == 0:
                    curator_snapshot = self.playbook
                    stats = get_playbook_stats(curator_snapshot)

                    # Aggregate reflections across the batch into one markdown-delimited
                    # block. Includes per-sample metadata so curator can attribute each
                    # reflection to its source question.
                    step_to_sample = {s: (smp, phase1_by_step[s]) for smp, s in zip(batch, batch_steps)}
                    reflection_blocks = []
                    for s in batch_steps:
                        smp, p1 = step_to_sample[s]
                        q = (smp.get('question') or '')[:300]
                        tgt = smp.get('target', '?')
                        ok = '✓' if p1.get('is_correct_pre') else '✗'
                        refl = p1.get('reflection_content') or '(no reflection)'
                        reflection_blocks.append(
                            f"### Sample step={s} ({ok} target={tgt})\n"
                            f"Question: {q}\n"
                            f"Reflection:\n{refl}"
                        )
                    aggregated_reflection = (
                        f"Batch of {len(batch_steps)} recent samples (steps {batch_steps[0]}-{batch_steps[-1]}). "
                        f"Make holistic edits that address patterns across the batch, not per-sample fixes.\n\n"
                        + "\n\n---\n\n".join(reflection_blocks)
                    )

                    # Aggregate context (drop duplicates/empty)
                    contexts = [smp.get('context','') for smp in batch if smp.get('context')]
                    aggregated_context = "\n\n".join(contexts) if contexts else ""

                    print(f"\n--- Running 1 Curator on aggregated batch (steps {batch_steps[0]}-{batch_steps[-1]}, {len(batch_steps)} reflections) ---")
                    # Discard curator's returned next_global_id and updated_playbook —
                    # curate() already calls apply_curator_operations internally but its
                    # result is against the snapshot we pass in. We re-apply ops to the
                    # live self.playbook below so the single authoritative bump comes from
                    # the explicit call. (Same pattern as the old parallel code.)
                    _, _, ops, _ = self.curator.curate(
                        current_playbook=curator_snapshot,
                        recent_reflection=aggregated_reflection,
                        question_context=aggregated_context,
                        current_step=batch_steps[-1], total_samples=original_total,
                        token_budget=token_budget, playbook_stats=stats,
                        use_ground_truth=not no_ground_truth,
                        use_json_mode=use_json_mode,
                        call_id=f"train_e_{epoch}_b{batch_start}",
                        log_dir=log_dir,
                        next_global_id=self.next_global_id,
                    )
                    all_ops = ops or []
                    if all_ops:
                        self.playbook, self.next_global_id, _ = apply_curator_operations(
                            self.playbook, all_ops, self.next_global_id,
                            current_step=batch_steps[-1],
                        )
                        print(f"  Applied {len(all_ops)} ops from aggregated curator")

                    if self.use_bulletpoint_analyzer and self.bulletpoint_analyzer:
                        print(f"  Running BulletpointAnalyzer (threshold={self.bulletpoint_analyzer_threshold})...")
                        self.playbook = self.bulletpoint_analyzer.analyze(
                            playbook=self.playbook,
                            threshold=self.bulletpoint_analyzer_threshold,
                            merge=True,
                            log_dir=log_dir,
                            call_id_prefix=f"analyzer_e{epoch}_b{batch_start}",
                        )

                    # PRUNE: deterministic lifecycle archive (fork §3.2 grow-and-refine)
                    self.playbook, archived = prune_playbook(
                        self.playbook, current_step=batch_steps[-1],
                        max_active_bullets_per_section=config_params.get('prune_max_active_bullets_per_section', 40),
                        warmup_window=config_params.get('prune_warmup_window', 50),
                        min_observations=config_params.get('prune_min_observations', 3),
                    )
                    if archived:
                        preview = archived[:5]
                        more = "…" if len(archived) > 5 else ""
                        print(f"  Archived {len(archived)} bullets: {preview}{more}")

                # PHASE 3: parallel post-curate generation against live playbook
                with ThreadPoolExecutor(max_workers=len(batch)) as exe:
                    futs = {
                        exe.submit(
                            self._train_sample_phase3,
                            sample, self.playbook,
                            f"train_e_{epoch}_s_{step}", log_dir, use_json_mode,
                        ): step
                        for sample, step in zip(batch, batch_steps)
                    }
                    phase3_by_step = {futs[f]: f.result() for f in as_completed(futs)}

                # Collect metrics + checkpointing per step in batch
                for sample, step in zip(batch, batch_steps):
                    p1 = phase1_by_step[step]
                    post_train_answer = phase3_by_step[step]
                    target = sample.get("target", "")

                    epoch_answers_pre_train.append(p1['pre_train_answer'])
                    epoch_targets_pre_train.append(target)
                    epoch_answers_post_train.append(post_train_answer)
                    epoch_targets_post_train.append(target)

                    pre_train_post_train_results.append({
                        "epoch": epoch, "step": step, "target": target,
                        "pre_train_result": {
                            "final_answer": p1['pre_train_answer'],
                            "is_correct": p1['is_correct_pre'],
                            "playbook_num_tokens": count_tokens(self.playbook),
                            "playbook_length": len(self.playbook),
                        },
                        "post_train_result": {
                            "final_answer": post_train_answer,
                            "is_correct": data_processor.answer_is_correct(post_train_answer, target),
                            "playbook_num_tokens": count_tokens(self.playbook),
                            "playbook_length": len(self.playbook),
                        },
                    })

                    if step % save_steps == 0:
                        intermediate_path = os.path.join(
                            playbook_dir, f"epoch_{epoch}_step_{step}_playbook.txt"
                        )
                        with open(intermediate_path, "w", encoding="utf-8") as f:
                            f.write(self.playbook)

                    if step % eval_steps == 0:
                        print(f"\n{'='*40}\nEVALUATION AT EPOCH {epoch}, STEP {step}\n{'='*40}")
                        pre_acc = data_processor.evaluate_accuracy(
                            epoch_answers_pre_train, epoch_targets_pre_train
                        )
                        post_acc = data_processor.evaluate_accuracy(
                            epoch_answers_post_train, epoch_targets_post_train
                        )
                        val_results: Dict[str, Any] = {}
                        val_error_log: Dict[str, Any] = {}
                        if val_samples:
                            val_results, val_error_log = evaluate_test_set(
                                data_processor, self.generator, render_minimal_playbook(self.playbook),
                                val_samples, self.max_tokens, log_dir,
                                max_workers=test_workers, use_json_mode=use_json_mode,
                            )
                        result = {
                            "epoch": epoch, "step": step,
                            "train_result": {
                                "pre_train_accuracy": pre_acc,
                                "post_train_accuracy": post_acc,
                            },
                            "val_result": val_results,
                            "playbook_num_tokens": count_tokens(self.playbook),
                            "playbook_length": len(self.playbook),
                            "playbook_stats": get_playbook_stats(self.playbook),
                        }
                        results.append(result)
                        error_logs.append({
                            "epoch": epoch, "step": step,
                            "val_results": val_results, "error_log": val_error_log,
                        })
                        if val_results and val_results.get("accuracy", 0) > best_accuracy:
                            best_accuracy = val_results["accuracy"]
                            self.best_playbook = self.playbook
                            print(f"🎉 New best accuracy: {best_accuracy:.3f}")
                        self._write_offline_progress(
                            save_path,
                            best_accuracy,
                            results,
                            error_logs,
                        )

            # End-of-epoch playbook
            epoch_playbook_path = os.path.join(
                playbook_dir, f"epoch_{epoch}_final_playbook.txt"
            )
            with open(epoch_playbook_path, "w", encoding="utf-8") as f:
                f.write(self.playbook)

        # Final saves (mirror _offline_train)
        self._write_offline_progress(
            save_path,
            best_accuracy,
            results,
            error_logs,
        )
        with open(os.path.join(save_path, "pre_train_post_train_results.json"), "w", encoding="utf-8") as f:
            json.dump(pre_train_post_train_results, f, indent=2, ensure_ascii=False)
        final_playbook_path = os.path.join(save_path, "final_playbook.txt")
        with open(final_playbook_path, "w", encoding="utf-8") as f:
            f.write(self.playbook)
        with open(os.path.join(save_path, "final_playbook_clean.txt"), "w", encoding="utf-8") as f:
            f.write(render_minimal_playbook(self.playbook))
        best_playbook_path = os.path.join(save_path, "best_playbook.txt")
        with open(best_playbook_path, "w", encoding="utf-8") as f:
            f.write(self.best_playbook)
        with open(os.path.join(save_path, "best_playbook_clean.txt"), "w", encoding="utf-8") as f:
            f.write(render_minimal_playbook(self.best_playbook))

        return self._build_offline_training_results(
            best_accuracy,
            results=results,
            pre_train_post_train_results=pre_train_post_train_results,
        )

    def _offline_train(
        self,
        train_samples: List[Dict[str, Any]],
        val_samples: List[Dict[str, Any]],
        data_processor,
        config: Dict[str, Any],
        save_path: str,
        usage_log_path: str,
        playbook_dir: str,
        log_dir: str
    ) -> Dict[str, Any]:
        """
        Run offline training
        
        Args:
            train_samples: List of training samples
            val_samples: List of validation samples
            data_processor: Data processor instance for the task
            config: Configuration dictionary
            save_path: Path to save results
            usage_log_path: Path for bullet usage logging
            playbook_dir: Directory for intermediate playbooks
            log_dir: Directory for detailed logs
            
        Returns:
            Dictionary with training results
        """
        # Extract configuration using helper
        config_params = self._extract_config_params(config)
        task_name = config_params['task_name']
        num_epochs = config_params['num_epochs']
        eval_steps = config_params['eval_steps']
        save_steps = config_params['save_steps']
        test_workers = config_params['test_workers']
        use_json_mode = config_params['use_json_mode']
        curator_frequency = config_params['curator_frequency']
        batch_size = config_params.get('batch_size', 1)

        if batch_size and batch_size > 1:
            return self._offline_train_batched(
                train_samples=train_samples,
                val_samples=val_samples,
                data_processor=data_processor,
                config=config,
                save_path=save_path,
                usage_log_path=usage_log_path,
                playbook_dir=playbook_dir,
                log_dir=log_dir,
                batch_size=batch_size,
            )

        # Initialize tracking
        results = []
        pre_train_post_train_results = []
        error_logs = []
        best_accuracy = 0.0
        self.best_playbook = self.playbook
        skip_n = config_params.get('skip_first_train_samples', 0)
        start_epoch = config_params.get('resume_epoch', 1)

        original_total = len(train_samples)
        if skip_n > 0:
            print(f"Resume: skipping first {skip_n} train samples")
            train_samples = train_samples[skip_n:]

        print(f"Epochs: {start_epoch}..{start_epoch + num_epochs - 1}")
        print(f"Train samples per epoch: {len(train_samples)} (of original {original_total})")
        print(f"Val samples: {len(val_samples)}")
        print(f"Curator frequency: every {curator_frequency} steps")
        print(f"Evaluation frequency: every {eval_steps} steps\n")

        # Training loop
        for epoch in range(start_epoch, start_epoch + num_epochs):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch}/{num_epochs}")
            print(f"{'='*60}")

            epoch_answers_pre_train = []
            epoch_targets_pre_train = []
            epoch_answers_post_train = []
            epoch_targets_post_train = []

            epoch_offset = (epoch - 1) * original_total
            for step, task_dict in enumerate(train_samples):
                step += 1 + skip_n + epoch_offset
                print(f"\n--- Step {step}/{original_total} ---")
                
                target = task_dict.get("target", "")
                
                # Use helper method for training single sample
                pre_train_answer, post_train_answer, tracking_dict = self._train_single_sample(
                    task_dict=task_dict,
                    data_processor=data_processor,
                    step_id=f"train_e_{epoch}_s_{step}",
                    epoch=epoch,
                    step=step,
                    usage_log_path=usage_log_path,
                    log_dir=log_dir,
                    config_params=config_params,
                    total_samples=original_total
                )
                
                # Collect answers for accuracy calculation
                epoch_answers_pre_train.append(pre_train_answer)
                epoch_targets_pre_train.append(target)
                epoch_answers_post_train.append(post_train_answer)
                epoch_targets_post_train.append(target)
                
                # Track pre-train and post-train results
                pre_train_post_train_result = {
                    "epoch": epoch,
                    "step": step,
                    "target": target,
                    **tracking_dict
                }
                pre_train_post_train_results.append(pre_train_post_train_result)
                
                # Save intermediate playbook
                if step % save_steps == 0:
                    intermediate_path = os.path.join(
                        playbook_dir, f"epoch_{epoch}_step_{step}_playbook.txt"
                    )
                    with open(intermediate_path, "w") as f:
                        f.write(self.playbook)
                
                # Periodic evaluation
                if step % eval_steps == 0:
                    print(f"\n{'='*40}")
                    print(f"EVALUATION AT EPOCH {epoch}, STEP {step}")
                    print(f"{'='*40}")
                    
                    # Compute training accuracies
                    pre_train_accuracy = data_processor.evaluate_accuracy(
                        epoch_answers_pre_train, epoch_targets_pre_train
                    )
                    post_train_accuracy = data_processor.evaluate_accuracy(
                        epoch_answers_post_train, epoch_targets_post_train
                    )
                    
                    # Validation evaluation
                    val_results = {}
                    val_error_log = {}
                    if val_samples:
                        val_results, val_error_log = evaluate_test_set(
                            data_processor, self.generator, render_minimal_playbook(self.playbook),
                            val_samples, self.max_tokens, log_dir,
                            max_workers=test_workers, use_json_mode=use_json_mode
                        )
                    
                    result = {
                        "epoch": epoch,
                        "step": step,
                        "train_result": {
                            "pre_train_accuracy": pre_train_accuracy,
                            "post_train_accuracy": post_train_accuracy
                        },
                        "val_result": val_results,
                        "playbook_num_tokens": count_tokens(self.playbook),
                        "playbook_length": len(self.playbook),
                        "playbook_stats": get_playbook_stats(self.playbook)
                    }
                    results.append(result)
                    error_logs.append({
                        "epoch": epoch,
                        "step": step,
                        "val_results": val_results,
                        "error_log": val_error_log
                    })

                    # Track best playbook
                    if val_results:
                        acc = val_results["accuracy"]
                        if acc > best_accuracy:
                            best_accuracy = acc
                            self.best_playbook = self.playbook
                            print(f"🎉 New best accuracy: {best_accuracy:.3f}")
                    
                    # Save results
                    self._write_offline_progress(
                        save_path,
                        best_accuracy,
                        results,
                        error_logs,
                    )
            
            # End of epoch - save final playbook
            epoch_playbook_path = os.path.join(
                playbook_dir, f"epoch_{epoch}_final_playbook.txt"
            )
            with open(epoch_playbook_path, "w") as f:
                f.write(self.playbook)

        # Save training results
        self._write_offline_progress(
            save_path,
            best_accuracy,
            results,
            error_logs,
        )

        pre_train_post_train_results_path = os.path.join(save_path, "pre_train_post_train_results.json")
        with open(pre_train_post_train_results_path, "w", encoding="utf-8") as f:
            json.dump(pre_train_post_train_results, f, indent=2, ensure_ascii=False)
        
        # Save final playbook
        final_playbook_path = os.path.join(save_path, f"final_playbook.txt")
        with open(final_playbook_path, "w") as f:
            f.write(self.playbook)
        with open(os.path.join(save_path, "final_playbook_clean.txt"), "w", encoding="utf-8") as f:
            f.write(render_minimal_playbook(self.playbook))

        # Save best playbook
        best_playbook_path = os.path.join(save_path, f"best_playbook.txt")
        with open(best_playbook_path, "w") as f:
            f.write(self.best_playbook)
        with open(os.path.join(save_path, "best_playbook_clean.txt"), "w", encoding="utf-8") as f:
            f.write(render_minimal_playbook(self.best_playbook))
        
        print(f"\n{'='*60}")
        print(f"OFFLINE TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Best Validation Accuracy: {best_accuracy:.3f}")
        print(f"{'='*60}\n")

        return self._build_offline_training_results(
            best_accuracy,
            results=results,
            pre_train_post_train_results=pre_train_post_train_results,
        )

    
    def test(
        self,
        test_samples: List[Dict[str, Any]],
        data_processor,
        playbook,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run testing with the playbook (backward compatibility wrapper).
        
        Args:
            test_samples: List of test samples
            data_processor: Data processor instance for the task
            playbook: Playbook to be used for generator
            config: Configuration dictionary
            
        Returns:
            Dictionary with test results
        """
        # Temporarily set the playbook
        old_playbook = self.playbook
        self.playbook = playbook
        
        # Use the run method
        results = self.run(
            mode='eval_only',
            test_samples=test_samples,
            data_processor=data_processor,
            config=config
        )
        
        # Restore old playbook
        self.playbook = old_playbook
        
        # Return in the old format for backward compatibility
        return {
            "test_results": results['test_results'],
            "error_log": results.get('test_error_log', {}),
            "playbook": playbook
        }
    
    def _online_train_and_test(
        self,
        test_samples: List[Dict[str, Any]],
        data_processor,
        config: Dict[str, Any],
        save_path: str,
        usage_log_path: str,
        playbook_dir: str,
        log_dir: str
    ) -> Dict[str, Any]:
        """
        Run online training and testing
        
        Args:
            test_samples: List of samples to train and test on
            data_processor: Data processor instance for the task
            config: Configuration dictionary
            save_path: Path to save results
            usage_log_path: Path for bullet usage logging
            playbook_dir: Directory for intermediate playbooks
            log_dir: Directory for detailed logs
            
        Returns:
            Dictionary with training results, test results, and final playbook
        """
        # Extract configuration using helper
        config_params = self._extract_config_params(config)
        num_epochs = config_params['num_epochs']
        
        # Validate configuration
        if num_epochs != 1:
            raise ValueError(f"online_train_and_test requires num_epochs=1, got {num_epochs}")
        
        # Extract additional parameters
        curator_frequency = config_params['curator_frequency']
        task_name = config_params['task_name']
        save_steps = config_params['save_steps']
        use_json_mode = config_params['use_json_mode']
        test_workers = config_params['test_workers']
        online_eval_frequency = config.get('online_eval_frequency', 100)  # Get from config
        
        # Initialize tracking
        train_results = []
        pre_train_post_train_results = []
        
        # Test tracking - accumulate across all windows
        correct_count_sample_based = 0
        correct_count = 0
        total_count = 0
        all_test_errors = []
        window_test_results = []
        print(f"Total samples: {len(test_samples)}")
        print(f"Window size: {online_eval_frequency}")
        print(f"Number of windows: {(len(test_samples) + online_eval_frequency - 1) // online_eval_frequency}")
        print(f"Curator frequency: every {curator_frequency} steps")
        
        # Split samples into windows
        num_windows = (len(test_samples) + online_eval_frequency - 1) // online_eval_frequency
        
        epoch = 1  # Always 1 epoch
        global_step = 0
        
        for window_idx in range(num_windows):
            start_idx = window_idx * online_eval_frequency
            end_idx = min((window_idx + 1) * online_eval_frequency, len(test_samples))
            window_samples = test_samples[start_idx:end_idx]
            
            print(f"\n{'='*60}")
            print(f"WINDOW {window_idx + 1}/{num_windows}")
            print(f"Samples {start_idx} to {end_idx - 1}")
            print(f"{'='*60}")
            
            # =================================================================
            # STEP 1: TEST on window with current playbook (before training)
            # =================================================================
            print(f"\n--- Testing window {window_idx + 1} with current playbook ---")
            
            # Use evaluate_test_set for parallel evaluation
            window_test_results_dict, window_test_error_log = evaluate_test_set(
                data_processor,
                self.generator,
                render_minimal_playbook(self.playbook),
                window_samples,
                self.max_tokens,
                log_dir,
                max_workers=test_workers,
                use_json_mode=use_json_mode
            )
            
            # Extract results
            window_accuracy = window_test_results_dict['accuracy']
            window_correct = window_test_results_dict['correct']
            window_total = window_test_results_dict['total']
            correct_count_sample_based += window_correct
            correct_count += window_accuracy * window_total
            total_count += window_total
            
            # Add errors with window and global index information
            for error in window_test_error_log['errors']:
                all_test_errors.append({
                    "window": window_idx + 1,
                    "global_index": start_idx + error['index'],
                    "prediction": error['prediction'],
                    "ground_truth": error['ground_truth']
                })
            
            window_test_results.append({
                "window": window_idx + 1,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "window_accuracy": window_accuracy,
                "window_correct": window_correct,
                "window_total": window_total
            })
            
            # Calculate cumulative test accuracy so far
            cumulative_test_accuracy = correct_count / total_count
            
            print(f"Window {window_idx + 1} test accuracy: {window_accuracy:.3f}")
            print(f"Cumulative test accuracy so far: {cumulative_test_accuracy:.3f} "
                  f"({total_count} samples)")
            
            # =================================================================
            # STEP 2: TRAIN on window (same as offline_train)
            # =================================================================
            print(f"\n--- Training on window {window_idx + 1} ---")
            
            epoch_answers_pre_train = []
            epoch_targets_pre_train = []
            epoch_answers_post_train = []
            epoch_targets_post_train = []
            
            for local_step, task_dict in enumerate(window_samples):
                global_step += 1
                local_step += 1
                
                print(f"\n--- Window {window_idx + 1}, Step {local_step}/{len(window_samples)} "
                      f"(Global step {global_step}) ---")
                
                target = task_dict.get("target", "")
                
                # Use helper method for training single sample
                pre_train_answer, post_train_answer, tracking_dict = self._train_single_sample(
                    task_dict=task_dict,
                    data_processor=data_processor,
                    step_id=f"online_train_s_{global_step}",
                    epoch=epoch,
                    step=global_step,
                    usage_log_path=usage_log_path,
                    log_dir=log_dir,
                    config_params=config_params,
                    total_samples=len(test_samples)
                )
                
                # Collect answers for accuracy calculation
                epoch_answers_pre_train.append(pre_train_answer)
                epoch_targets_pre_train.append(target)
                epoch_answers_post_train.append(post_train_answer)
                epoch_targets_post_train.append(target)
                
                # Track pre-train and post-train results
                pre_train_post_train_result = {
                    "window": window_idx + 1,
                    "global_step": global_step,
                    "target": target,
                    **tracking_dict
                }
                pre_train_post_train_results.append(pre_train_post_train_result)
                
                # Save intermediate playbook
                if global_step % save_steps == 0:
                    intermediate_path = os.path.join(
                        playbook_dir, f"step_{global_step}_playbook.txt"
                    )
                    with open(intermediate_path, "w") as f:
                        f.write(self.playbook)
            
            # End of window - compute training accuracies for this window
            pre_train_accuracy = data_processor.evaluate_accuracy(
                epoch_answers_pre_train, epoch_targets_pre_train
            )
            post_train_accuracy = data_processor.evaluate_accuracy(
                epoch_answers_post_train, epoch_targets_post_train
            )
            
            window_train_result = {
                "window": window_idx + 1,
                "global_step": global_step,
                "train_result": {
                    "pre_train_accuracy": pre_train_accuracy,
                    "post_train_accuracy": post_train_accuracy
                },
                "cumulative_test_accuracy": cumulative_test_accuracy,
                "playbook_num_tokens": count_tokens(self.playbook),
                "playbook_length": len(self.playbook),
                "playbook_stats": get_playbook_stats(self.playbook)
            }
            train_results.append(window_train_result)
            
            print(f"\nWindow {window_idx + 1} training complete:")
            print(f"  Pre-train accuracy: {pre_train_accuracy:.3f}")
            print(f"  Post-train accuracy: {post_train_accuracy:.3f}")
            
            # Save window playbook
            window_playbook_path = os.path.join(
                playbook_dir, f"window_{window_idx + 1}_final_playbook.txt"
            )
            with open(window_playbook_path, "w") as f:
                f.write(self.playbook)
        
        # All windows complete
        print(f"\n{'='*60}")
        print(f"ONLINE TRAIN AND TEST COMPLETE")
        print(f"{'='*60}")
        
        # Calculate final cumulative test accuracy
        assert total_count == len(test_samples)
        final_test_accuracy = correct_count / total_count
        
        test_results = {
            "accuracy": final_test_accuracy,
            "correct": correct_count_sample_based,
            "total": total_count,
            "window_results": window_test_results
        }
        
        test_error_log = {
            "accuracy": final_test_accuracy,
            "errors": all_test_errors
        }

        # Save test results
        test_results_path = os.path.join(save_path, "test_results.json")
        with open(test_results_path, "w", encoding="utf-8") as f:
            json.dump({
                "test_accuracy": final_test_accuracy,
                "test_results": test_results,
                "test_error_log": test_error_log
            }, f, indent=2, ensure_ascii=False)

        # Save training results (per window)
        train_results_path = os.path.join(save_path, "train_results.json")
        with open(train_results_path, "w", encoding="utf-8") as f:
            json.dump({"train_results": train_results}, f, indent=2, ensure_ascii=False)

        # Save pre-train/post-train results
        pre_train_post_train_results_path = os.path.join(save_path, "pre_train_post_train_results.json")
        with open(pre_train_post_train_results_path, "w", encoding="utf-8") as f:
            json.dump(pre_train_post_train_results, f, indent=2, ensure_ascii=False)
        
        # Save final playbook
        final_playbook_path = os.path.join(save_path, f"final_playbook.txt")
        with open(final_playbook_path, "w") as f:
            f.write(self.playbook)
        with open(os.path.join(save_path, "final_playbook_clean.txt"), "w", encoding="utf-8") as f:
            f.write(render_minimal_playbook(self.playbook))

        print(f"\n{'='*60}")
        print(f"ONLINE TRAINING AND TESTING COMPLETE")
        print(f"{'='*60}")
        print(f"Final Test Accuracy: {final_test_accuracy:.3f}")
        print(f"{'='*60}\n")
        
        return {
            "accuracy": final_test_accuracy,
            "correct": correct_count_sample_based,
            "total": total_count,
        }
