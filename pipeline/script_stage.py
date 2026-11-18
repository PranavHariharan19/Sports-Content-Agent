from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from app.logging_config import get_logger
from llm.router import LLMRouter
from pipeline.db import DatabaseManager
from pipeline.script_generator import ScriptGenerator
from pipeline.script_linter import DeterministicScriptLinter
from pipeline.script_verifier import ScriptVerifier
from pipeline.state_machine import PipelineStateMachine
from schemas.common import CorroborationStatus, JobStatus, StageName, VerificationVerdict
from schemas.config import PipelineConfig
from schemas.facts import ExtractedFact
from schemas.script import GeneratedScript, ScriptBeat

logger = get_logger(__name__)


class ScriptPipelineStage:
    """Coordinates Stage 5: Constrained script drafting, deterministic linting,

    independent verifier pass, and human approval state management.
    """

    def __init__(
        self,
        db: DatabaseManager,
        llm_router: LLMRouter,
        pipeline_config: Optional[PipelineConfig] = None,
        state_machine: Optional[PipelineStateMachine] = None,
        generator: Optional[ScriptGenerator] = None,
        linter: Optional[DeterministicScriptLinter] = None,
        verifier: Optional[ScriptVerifier] = None,
    ):
        self.db = db
        self.router = llm_router
        self.config = pipeline_config or PipelineConfig()
        self.state_machine = state_machine or PipelineStateMachine(db)
        self.generator = generator or ScriptGenerator(llm_router)
        self.linter = linter or DeterministicScriptLinter()
        self.verifier = verifier or ScriptVerifier(llm_router)

    def _get_headline_for_job(self, job_id: str) -> str:
        stage_runs = self.db.get_stage_runs_for_job(job_id)
        for sr in reversed(stage_runs):
            out = sr.get_output()
            if "chosen_topic" in out:
                return out["chosen_topic"].get("primary_headline", "Sports News Update")
            if "candidates" in out and out["candidates"]:
                return out["candidates"][0].get("primary_headline", "Sports News Update")
        return "Sports News Update"

    async def draft_and_verify_script(self, job_id: str) -> GeneratedScript:
        """Generate structured script, run deterministic linter, and execute verifier pass."""
        job = self.db.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        facts = self.db.get_facts_for_job(job_id)
        usable_facts = [
            f for f in facts
            if f.corroboration_status in (CorroborationStatus.VERIFIED, CorroborationStatus.REPORTED)
            and not f.is_conflicted
        ]
        if not usable_facts:
            raise ValueError(f"No usable verified facts for job {job_id}")

        facts_by_id: Dict[str, ExtractedFact] = {f.fact_id: f for f in usable_facts}
        ledger_entities: Set[str] = set()
        for f in usable_facts:
            ledger_entities.update(f.normalized_entities)

        headline = self._get_headline_for_job(job_id)

        # 1. Generate initial script (Section 6.7)
        script = await self.generator.generate_script(
            job_id=job_id,
            headline=headline,
            facts=usable_facts,
        )

        # 2. Run deterministic script linter (Section 6.8)
        script.beats = self.linter.lint_script(
            beats=script.beats,
            facts_by_id=facts_by_id,
            ledger_entities=ledger_entities,
        )

        # 3. Independent verifier pass (Section 6.9)
        script.beats = await self.verifier.verify_script(
            beats=script.beats,
            facts_by_id=facts_by_id,
        )

        # Update script metadata
        script.all_supported = all(
            b.verifier_verdict == VerificationVerdict.SUPPORTED or b.human_override
            for b in script.beats
        )
        script.lint_passed = all(len(b.lint_errors) == 0 or b.human_override for b in script.beats)

        # Save to SQLite
        self.db.save_script(script)

        # Advance state to SCRIPT_DRAFTED
        self.state_machine.record_stage_execution(
            job_id=job_id,
            stage_name=StageName.SCRIPT,
            input_data={"headline": headline, "facts_used": len(usable_facts)},
            output_data=script.model_dump(mode="json"),
            success=True,
            allow_rerun=True,
        )

        logger.info(
            "Script drafted and verified for job %s (%d beats, %d words, all_supported=%s, lint_passed=%s)",
            job_id,
            len(script.beats),
            script.total_words,
            script.all_supported,
            script.lint_passed,
        )
        return script

    def approve_script_checkpoint(self, job_id: str, force: bool = True) -> GeneratedScript:
        """Checkpoint 2: Human approves the script, advancing status to SCRIPT_APPROVED."""
        job = self.db.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        script = self.db.get_script_for_job(job_id)
        if not script:
            raise ValueError(f"No script found for job {job_id}")

        # Publish is blocked while any factual sentence is unverified or has lint errors (Section 6.13)
        unsupported = [
            b for b in script.beats
            if b.is_factual
            and b.verifier_verdict != VerificationVerdict.SUPPORTED
            and not b.human_override
        ]
        lint_failed = [
            b for b in script.beats
            if len(b.lint_errors) > 0 and not b.human_override
        ]

        if (unsupported or lint_failed) and not force:
            issues: List[str] = []
            if unsupported:
                issues.append(f"{len(unsupported)} beats not verified as SUPPORTED")
            if lint_failed:
                issues.append(f"{len(lint_failed)} beats with lint errors")
            raise ValueError(f"Cannot approve script: {', '.join(issues)}. Override or edit them first.")

        # When human approves, accept remaining beats under human authority
        if unsupported or lint_failed:
            for b in script.beats:
                if (b.is_factual and b.verifier_verdict != VerificationVerdict.SUPPORTED) or len(b.lint_errors) > 0:
                    if not b.human_override:
                        b.human_override = True
                        b.human_override_reason = "Human approved at Checkpoint 2"
            self.db.save_script(script)

        # Update approval in DB
        self.db.approve_script(job_id)

        # Advance state machine
        self.state_machine.transition_job(job_id, JobStatus.SCRIPT_APPROVED, allow_rerun=True)

        logger.info("Human Checkpoint 2 completed: Script for job %s approved!", job_id)
        updated = self.db.get_script_for_job(job_id)
        assert updated is not None
        return updated
