"""
Stage 1: Conditioned Failure Extractor (CS-US -> ReflexRule).
Parses learning phase interactions for unconditioned failure stimuli (u_{US})
and extracts structured contrastive inhibitory rules r_i.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from .schemas import ReflexRule
from .embedder import SemanticTriggerEmbedder

logger = logging.getLogger(__name__)

# Patterns indicative of unconditioned stimuli (u_{US})
FAILURE_PATTERNS = [
    r"(?i)permission denied",
    r"(?i)access denied",
    r"(?i)rate limit exceeded",
    r"(?i)429 too many requests",
    r"(?i)deprecated",
    r"(?i)connection refused",
    r"(?i)timeout expired",
    r"(?i)segmentation fault",
    r"(?i)corrupted",
    r"(?i)syntaxerror",
    r"(?i)typeerror",
    r"(?i)valueerror",
    r"(?i)security error",
    r"(?i)prohibited",
    r"(?i)do not use",
    r"(?i)stop using",
    r"(?i)never call",
    r"(?i)confused",
    r"(?i)too technical",
    r"(?i)explain like i'm a beginner",
    r"(?i)fatal:",
    r"(?i)error 500",
]


class ConditionedFailureExtractor:
    """
    Extracts ReflexRule tuples:
    r_i = (e_{CS, i}, P_{CS, i}, a_{inhibit, i}, V_{inhibit, i}, a_{safe, i})
    from logged interaction episodes.
    """

    def __init__(self, embedder: Optional[SemanticTriggerEmbedder] = None):
        self.embedder = embedder or SemanticTriggerEmbedder()

    def detect_unconditioned_stimulus(self, text: str) -> Optional[str]:
        """
        Scans observation/feedback text for negative unconditioned stimulus cues.
        """
        for pat in FAILURE_PATTERNS:
            match = re.search(pat, text)
            if match:
                return match.group(0)
        return None

    def extract_from_scenario(
        self,
        scenario_id: str,
        domain: str,
        learning_turns: List[Dict[str, str]],
        explicit_trigger: Optional[str] = None,
        inhibited_action: Optional[str] = None,
        safe_alternative: Optional[str] = None,
        inhibited_tokens: Optional[List[str]] = None,
    ) -> ReflexRule:
        """
        Synthesizes a ReflexRule from learning turns and ground truth annotations.
        """
        # Scan turns for failure cue
        detected_cue = "Unconditioned execution failure"
        extracted_trigger = explicit_trigger or ""

        for turn in learning_turns:
            content = turn.get("content", "")
            role = turn.get("role", "")
            if role in ["user", "environment", "system"]:
                cue = self.detect_unconditioned_stimulus(content)
                if cue:
                    detected_cue = cue
            if role == "user" and not extracted_trigger:
                extracted_trigger = content

        if not extracted_trigger and learning_turns:
            extracted_trigger = learning_turns[0].get("content", "General trigger")

        inh_action = inhibited_action or "default_penalized_action"
        safe_alt = safe_alternative or "default_safe_alternative"

        # Generate inhibited token sequence candidates
        tokens = inhibited_tokens or []
        if not tokens and inh_action:
            tokens = [inh_action, inh_action.lower(), inh_action.upper()]
            # Split tool or path prefixes
            subtokens = re.split(r"[/._\s\-]+", inh_action)
            for st in subtokens:
                if len(st) > 2 and st not in tokens:
                    tokens.append(st)

        # Compute trigger embedding
        emb = self.embedder.embed(extracted_trigger)

        rule = ReflexRule(
            rule_id=f"rule_{scenario_id}",
            trigger_pattern=extracted_trigger,
            inhibited_action=inh_action,
            inhibited_tokens=tokens,
            safe_alternative=safe_alt,
            domain=domain,
            failure_cue=detected_cue,
            embedding=emb,
            metadata={"scenario_id": scenario_id},
        )
        return rule
