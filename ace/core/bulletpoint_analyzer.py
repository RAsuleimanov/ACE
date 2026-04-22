"""
BulletpointAnalyzer Component for ACE System

This component analyzes playbook bulletpoints for similarity and performs
intelligent deduplication and merging using embeddings and LLM.
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict

# Use the lifecycle-aware parse_playbook_line + status filter from the new format
from playbook_utils import parse_playbook_line, ACTIVE_BULLET_STATUSES

_CATALOG_NUM_RE = re.compile(r"^(\d+)\.")


def _catalog_docs_conflict(content_a: str, content_b: str) -> bool:
    """Block merge of catalog entries with different leading doc numbers."""
    m_a = _CATALOG_NUM_RE.match(content_a.strip())
    m_b = _CATALOG_NUM_RE.match(content_b.strip())
    if m_a and m_b:
        return m_a.group(1) != m_b.group(1)
    return False

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    DEDUP_AVAILABLE = True
except ImportError:
    DEDUP_AVAILABLE = False
    print("Warning: sentence-transformers or faiss not available for bulletpoint analysis.")
    print("Install with: pip install sentence-transformers faiss-cpu")


class BulletpointAnalyzer:
    """
    Bulletpoint analyzer for deduplication and merging of similar playbook entries.

    Uses sentence transformers for semantic similarity and LLM for intelligent merging.
    Default embedder is `BAAI/bge-m3` — hard-negative trained multilingual model,
    SOTA on MTEB-multilingual. Hard-negatives training crucial for distinguishing
    semantically-close but different items (e.g. "37. Выписка по детскому" vs
    "39. Выписка на английском"). Size ~2.2GB, dim=1024. Switch via
    `embedding_model_name`.
    """

    def __init__(
        self,
        client,
        model: str,
        max_tokens: int = 4096,
        embedding_model_name: str = 'BAAI/bge-m3',
        api_provider: str = 'openai',
        bm25_threshold: float = 0.0,
        block_cross_section: bool = True,
        reasoning: Optional[dict] = None,
        embedding_client=None,
    ):
        """Initialize the bulletpoint analyzer.

        `client` is used for LLM merge calls; `embedding_client` (if given) is
        used for /v1/embeddings when embedding_model_name starts with "api:".
        Falls back to `client` when embedding_client is None.
        """
        self.client = client
        self.embedding_client = embedding_client or client
        self.model = model
        self.max_tokens = max_tokens
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        self.api_provider = api_provider
        self.bm25_threshold = bm25_threshold
        self.block_cross_section = block_cross_section
        self.reasoning = reasoning

        if not self._uses_api_embeddings() and not DEDUP_AVAILABLE:
            print("⚠️  Bulletpoint analyzer initialized but dependencies not available")

    def _uses_api_embeddings(self) -> bool:
        """Return True when embeddings should be fetched from an API endpoint."""
        return self.embedding_model_name.startswith("api:")
    
    def _load_embedding_model(self):
        """Load sentence transformer model for embeddings."""
        if self._uses_api_embeddings():
            return
        if not DEDUP_AVAILABLE:
            raise RuntimeError("Cannot load local embedding model without sentence-transformers")
        if self.embedding_model is None:
            print(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
    
    def _parse_playbook(self, playbook: str) -> Tuple[List[str], List[Dict[str, Any]], Dict[int, int]]:
        """
        Parse playbook into lines, bullets, and mapping.
        
        Args:
            playbook: Playbook content as string
            
        Returns:
            Tuple of (original_lines, bullets, bullet_line_mapping)
        """
        lines = playbook.strip().split('\n')
        bullets = []
        bullet_line_mapping = {}
        
        for line_idx, line in enumerate(lines):
            parsed = parse_playbook_line(line)
            if parsed:
                # Skip archived bullets — they shouldn't compete for merges
                if parsed.get('status', 'active') not in ACTIVE_BULLET_STATUSES:
                    continue
                parsed['line_number'] = line_idx + 1
                parsed['original_line'] = line
                bullet_index = len(bullets)
                bullet_line_mapping[bullet_index] = line_idx
                bullets.append(parsed)

        return lines, bullets, bullet_line_mapping
    
    def _compute_embeddings(self, bullets: List[Dict[str, Any]]) -> np.ndarray:
        """
        Compute embeddings for all bullets.

        `embedding_model_name` prefix "api:" routes the call to the OpenAI-
        compatible embeddings endpoint (e.g. OpenRouter's /v1/embeddings) using
        the same client passed in for merges. Everything else runs locally via
        sentence-transformers. API mode avoids ~1.5 GB MPS allocation on Mac
        and costs ~$0.01 per full offline run — worth it when the local model
        is OOM'ing against Cursor/Chrome GPU pressure.
        """
        contents = [bullet['content'] for bullet in bullets]

        if self._uses_api_embeddings():
            model_id = self.embedding_model_name[4:]
            # Retry transient upstream failures (empty data, 5xx, timeouts).
            # The analyzer pass is optional — if we can't get embeddings, skip
            # merge this round and let the next batch re-try rather than killing
            # the whole training run.
            import random as _rand
            last_err = None
            for attempt in range(5):
                try:
                    resp = self.embedding_client.embeddings.create(model=model_id, input=contents)
                    if not resp.data:
                        raise ValueError("No embedding data received")
                    embeddings = np.array([e.embedding for e in resp.data], dtype=np.float32)
                    break
                except Exception as e:
                    last_err = e
                    sleep = 5 * (2 ** attempt) * _rand.uniform(0.8, 1.2)
                    print(f"  ⚠️  Embedding API error (attempt {attempt+1}/5): {type(e).__name__}: {e}. Retrying in {sleep:.1f}s")
                    import time as _t; _t.sleep(sleep)
            else:
                print(f"  ⚠️  Embedding API failed after 5 retries; skipping analyzer pass. Last error: {last_err}")
                # Return empty-shaped array; caller (_find_similar_groups) will return []
                return np.zeros((len(contents), 1), dtype=np.float32)
        else:
            if not DEDUP_AVAILABLE:
                raise RuntimeError("Cannot compute embeddings without sentence-transformers")
            self._load_embedding_model()
            embeddings = self.embedding_model.encode(
                contents, convert_to_numpy=True, show_progress_bar=False
            ).astype(np.float32)

        # Normalize for cosine similarity (both paths)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms
    
    def _compute_bm25_matrix(self, bullets: List[Dict[str, Any]]) -> np.ndarray:
        """Pairwise normalised BM25 similarity.

        Tokeniser is word-level (keeps digits — important for doc IDs like "37").
        Each row is normalised by its max so scores are comparable in [0, 1].
        """
        from rank_bm25 import BM25Okapi
        import re as _re

        def tok(text: str) -> List[str]:
            return _re.findall(r"\w+", text.lower(), flags=_re.UNICODE)

        corpus = [tok(b['content']) for b in bullets]
        if not any(corpus):
            return np.zeros((len(bullets), len(bullets)))

        bm25 = BM25Okapi(corpus)
        n = len(bullets)
        matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            if not corpus[i]:
                continue
            scores = np.asarray(bm25.get_scores(corpus[i]), dtype=np.float32)
            # Normalise by self-score (diagonal = perfect match with self).
            # matrix[i, j] ∈ [0, 1] = "fraction of i's perfect-self-match that
            # document j achieves". Practical threshold ~0.1 for meaningful
            # lexical overlap in short playbook bullets.
            self_score = scores[i]
            if self_score > 0:
                scores = scores / self_score
            matrix[i] = scores
        return matrix

    def _find_similar_groups(
        self,
        bullets: List[Dict[str, Any]],
        embeddings: np.ndarray,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Find merge-candidate groups via dense embedding similarity,
        optionally AND-gated by normalised BM25 if `self.bm25_threshold > 0`.
        """
        similarity_matrix = np.dot(embeddings, embeddings.T)
        bm25_matrix = None
        if self.bm25_threshold > 0:
            bm25_matrix = self._compute_bm25_matrix(bullets)

        duplicate_groups = []
        visited = set()

        for i in range(len(bullets)):
            if i in visited:
                continue

            similar_indices = []
            for j in range(i + 1, len(bullets)):
                if similarity_matrix[i, j] < threshold:
                    continue
                if _catalog_docs_conflict(bullets[i].get('content', ''), bullets[j].get('content', '')):
                    continue
                if self.block_cross_section:
                    sec_i = bullets[i]['id'].split('-', 1)[0]
                    sec_j = bullets[j]['id'].split('-', 1)[0]
                    if sec_i != sec_j:
                        ISOLATED_SLUGS = {'пд', 'фо'}
                        if sec_i in ISOLATED_SLUGS or sec_j in ISOLATED_SLUGS:
                            continue
                if bm25_matrix is not None:
                    # AND-gate: max of (i→j) and (j→i) to be symmetric
                    bm25_sim = max(bm25_matrix[i, j], bm25_matrix[j, i])
                    if bm25_sim < self.bm25_threshold:
                        continue
                similar_indices.append(j)

            if similar_indices:
                group = [i] + similar_indices
                duplicate_groups.append({
                    'indices': group,
                    'bullets': [bullets[idx] for idx in group]
                })
                visited.update(group)

        return duplicate_groups
    
    def _merge_bullets_with_llm(self, bullets_group: List[Dict[str, Any]],
                                log_dir: Optional[str] = None,
                                call_id: str = "merge") -> Optional[Dict[str, Any]]:
        """Merge a group of similar bullets using LLM.

        Routes the call through `llm.timed_llm_call` so the merge prompt /
        response land in `llm_calls.jsonl` (auditability — previously the
        direct `client.chat.completions.create` call bypassed the logger).
        """
        if len(bullets_group) == 1:
            return bullets_group[0]

        # Prepare prompt for LLM
        bullets_text = "\n".join([
            f"{i+1}. [{b['id']}] helpful={b['helpful']} harmful={b['harmful']} :: {b['content']}"
            for i, b in enumerate(bullets_group)
        ])

        # Calculate combined helpful/harmful counts
        total_helpful = sum(b['helpful'] for b in bullets_group)
        total_harmful = sum(b['harmful'] for b in bullets_group)

        # Use first bullet's ID as base
        base_id = bullets_group[0]['id']

        prompt = f"""You are merging similar playbook bulletpoints into a single, comprehensive entry.

Given these similar bulletpoints:
{bullets_text}

Merge them into ONE bulletpoint that captures all important information while removing redundancy.

Requirements:
1. Keep the ID from the first entry: [{base_id}]
2. Use combined counts: helpful={total_helpful} harmful={total_harmful}
3. Combine the content to be comprehensive but concise
4. Output ONLY in this format: [{base_id}] helpful={total_helpful} harmful={total_harmful} :: [merged content]

Do NOT include any explanation, just output the merged bulletpoint."""

        try:
            from llm import timed_llm_call
            merged_content, _call_info = timed_llm_call(
                self.client, self.api_provider, self.model, prompt,
                role="analyzer_merge", call_id=call_id,
                max_tokens=self.max_tokens, log_dir=log_dir,
                reasoning=self.reasoning,
            )
            merged_content = (merged_content or "").strip()
            
            # Parse the merged bullet
            pattern = r'\[([^\]]+)\]\s+helpful=(\d+)\s+harmful=(\d+)\s+::\s+(.+)'
            match = re.match(pattern, merged_content)
            
            if match:
                bullet_id, helpful, harmful, content = match.groups()
                return {
                    'id': bullet_id,
                    'helpful': int(helpful),
                    'harmful': int(harmful),
                    'content': content.strip(),
                    'original_line': f"[{bullet_id}] helpful={helpful} harmful={harmful} :: {content.strip()}",
                    'is_merged': True,
                    'original_count': len(bullets_group)
                }
            else:
                print(f"⚠️  Failed to parse merged bullet, keeping first bullet from group")
                return bullets_group[0]
                
        except Exception as e:
            print(f"⚠️  Error merging bullets: {e}, keeping first bullet from group")
            return bullets_group[0]
    
    def analyze(
        self,
        playbook: str,
        threshold: float = 0.90,
        merge: bool = True,
        log_dir: Optional[str] = None,
        call_id_prefix: str = "merge",
    ) -> str:
        """
        Analyze and deduplicate/merge playbook bulletpoints.
        
        Args:
            playbook: Playbook content as string
            threshold: Similarity threshold for grouping (default: 0.90)
            merge: If True, merge similar bullets with LLM; if False, just deduplicate
            
        Returns:
            Processed playbook string
        """
        if not self._uses_api_embeddings() and not DEDUP_AVAILABLE:
            print("⚠️  Skipping bulletpoint analysis (dependencies not available)")
            return playbook
        
        # Parse playbook
        original_lines, bullets, bullet_line_mapping = self._parse_playbook(playbook)
        
        if len(bullets) == 0:
            return playbook
        
        print(f"Analyzing {len(bullets)} bulletpoints (threshold={threshold})...")
        
        # Compute embeddings
        embeddings = self._compute_embeddings(bullets)
        
        # Find similar groups
        duplicate_groups = self._find_similar_groups(bullets, embeddings, threshold)
        
        if len(duplicate_groups) == 0:
            print(f"No similar bulletpoints found at threshold {threshold}")
            return playbook
        
        print(f"Found {len(duplicate_groups)} groups of similar bulletpoints")
        
        # Create merge mapping
        merge_mapping = {}
        processed_indices = set()
        
        if merge:
            # Merge using LLM
            for group_idx, group in enumerate(duplicate_groups):
                indices = group['indices']
                group_bullets = group['bullets']
                
                print(f"  Merging group {group_idx + 1}: {len(group_bullets)} bullets -> 1")
                merged_bullet = self._merge_bullets_with_llm(
                    group_bullets, log_dir=log_dir,
                    call_id=f"{call_id_prefix}_g{group_idx + 1}",
                )
                
                if merged_bullet:
                    first_bullet_idx = indices[0]
                    merge_mapping[first_bullet_idx] = merged_bullet
                    processed_indices.update(indices)
        else:
            # Simple deduplication (keep first of each group)
            for group in duplicate_groups:
                indices = group['indices']
                # Keep first, mark others for removal
                processed_indices.update(indices[1:])
        
        # Reconstruct playbook
        output_lines = []
        
        for line_idx, original_line in enumerate(original_lines):
            # Check if this line is a bullet
            current_bullet_idx = None
            for bi in bullet_line_mapping:
                if bullet_line_mapping[bi] == line_idx:
                    current_bullet_idx = bi
                    break
            
            if current_bullet_idx is not None:
                # This is a bullet line
                if current_bullet_idx in merge_mapping:
                    # Use merged version
                    merged_bullet = merge_mapping[current_bullet_idx]
                    output_lines.append(merged_bullet['original_line'])
                elif current_bullet_idx in processed_indices:
                    # This bullet was merged into another, skip it
                    continue
                else:
                    # Keep original
                    output_lines.append(original_line)
            else:
                # Not a bullet line (headers, empty lines, etc.)
                output_lines.append(original_line)
        
        # Calculate statistics
        final_bullet_count = len(bullets) - len(processed_indices) + len(merge_mapping)
        removed_count = len(bullets) - final_bullet_count
        
        print(f"✓ Bulletpoint analysis complete: {len(bullets)} -> {final_bullet_count} "
              f"({removed_count} bullets merged/removed)")
        
        return '\n'.join(output_lines)
