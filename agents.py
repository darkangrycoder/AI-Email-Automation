"""
agents.py
─────────
Defines and runs the CrewAI 4-agent sequential email generation pipeline.

Pipeline flow  (Process.sequential, each task passes context to the next):

  PlannerAgent  ──►  WriterAgent  ──►  CriticAgent  ──►  RefinerAgent
      │                   │                  │                  │
  Blueprint          Draft email        Structured          Final email
  (5-step CoT)      (R-FSCoT)           critique           (send-ready)

Why 4 agents vs a single LLM call?
  • Each agent's system prompt is specialised: the Planner thinks structurally,
    the Writer thinks fluently, the Critic thinks analytically, the Refiner
    integrates. Specialised roles produce measurably better output than a
    single generalist call on email-generation tasks.
  • The Critic → Refiner step catches tone drift, missing facts, and structural
    issues that the initial writer misses — the same catch-and-fix loop used in
    professional editorial workflows.
  • CrewAI's context passing between tasks means the Refiner sees both the
    original draft AND the critique together, enabling targeted improvements.
"""

import re
import time

from config import MAX_RETRIES, RETRY_BASE_DELAY, INTER_SCENARIO_WAIT
from prompts import (
    SYSTEM_PERSONA, FEW_SHOT_EXAMPLES,
    PLANNER_ROLE, PLANNER_GOAL, PLANNER_BACKSTORY, PLANNER_TASK_TEMPLATE,
    WRITER_ROLE,  WRITER_GOAL,  WRITER_BACKSTORY,  WRITER_TASK_TEMPLATE,
    CRITIC_ROLE,  CRITIC_GOAL,  CRITIC_BACKSTORY,  CRITIC_TASK_TEMPLATE,
    REFINER_ROLE, REFINER_GOAL, REFINER_BACKSTORY, REFINER_TASK_TEMPLATE,
)


# ── Output cleaning ───────────────────────────────────────────────────────────

def clean_email_output(raw: str) -> str:
    """
    Strip reasoning blocks and normalise whitespace from LLM output.

    BUG FIX: Case-insensitive thinking-tag stripping.
    Qwen2.5 emits <Thinking>…</Thinking> (capital T), which the v1 regex
    r"<thinking>…</thinking>" failed to catch, leaking planning text into
    the final email. The re.IGNORECASE flag fixes this.

    Also handles DeepSeek-style <think>…</think> and generic <reasoning> tags.
    """
    cleaned = raw

    # Strip all known reasoning/planning block patterns
    for pat in [
        r"<thinking>.*?</thinking>",
        r"<think>.*?</think>",
        r"<reasoning>.*?</reasoning>",
        r"<reflection>.*?</reflection>",
    ]:
        cleaned = re.sub(pat, "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    # Strip any residual XML-like tags (e.g. <answer>, <output>)
    cleaned = re.sub(r"<[^>]{1,30}>", "", cleaned)

    # Collapse 3+ blank lines → 2 blank lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


# ── Agent factory ─────────────────────────────────────────────────────────────

def build_agents(llm_id: str) -> tuple:
    """
    Instantiate the four CrewAI agents backed by the given LiteLLM model string.

    Parameters
    ----------
    llm_id : str
        LiteLLM-compatible string, e.g. "ollama/qwen2.5:7b"

    Returns
    -------
    tuple: (planner, writer, critic, refiner) — four crewai.Agent instances
    """
    from crewai import Agent

    common_kwargs = dict(
        llm=llm_id,
        verbose=False,           # suppressed: app.py handles progress printing
        allow_delegation=False,  # no inter-agent delegation in sequential flow
        max_iter=2,              # max LLM retries per agent task
        max_retry_limit=2,
    )

    planner = Agent(
        role=PLANNER_ROLE,
        goal=PLANNER_GOAL,
        backstory=PLANNER_BACKSTORY,
        **common_kwargs,
    )

    writer = Agent(
        role=WRITER_ROLE,
        goal=WRITER_GOAL,
        backstory=WRITER_BACKSTORY,
        **common_kwargs,
    )

    critic = Agent(
        role=CRITIC_ROLE,
        goal=CRITIC_GOAL,
        backstory=CRITIC_BACKSTORY,
        **common_kwargs,
    )

    refiner = Agent(
        role=REFINER_ROLE,
        goal=REFINER_GOAL,
        backstory=REFINER_BACKSTORY,
        **common_kwargs,
    )

    return planner, writer, critic, refiner


# ── Per-scenario generation ───────────────────────────────────────────────────

def generate_email(planner, writer, critic, refiner, scenario: dict) -> str:
    """
    Run the 4-agent pipeline for a single scenario.

    Task descriptions use .format() to inject scenario values at call time,
    keeping the template strings in prompts.py clean and reusable.

    Parameters
    ----------
    scenario : dict
        One entry from scenarios.SCENARIOS

    Returns
    -------
    str  — the final polished email, or an error string on failure
    """
    from crewai import Task, Crew, Process

    intent       = scenario["intent"]
    key_facts    = "\n".join(f"- {f}" for f in scenario["key_facts"])
    tone         = scenario["tone"]

    # Task 1: Strategic planning
    plan_task = Task(
        description=PLANNER_TASK_TEMPLATE.format(
            intent=intent, key_facts=key_facts, tone=tone
        ),
        agent=planner,
        expected_output=(
            "A structured email blueprint between ---BLUEPRINT START--- and "
            "---BLUEPRINT END--- markers, covering: subject line options, "
            "paragraph plan, critical vocabulary, avoidance list, and risk note."
        ),
    )

    # Task 2: Draft writing (receives plan_task context automatically)
    write_task = Task(
        description=WRITER_TASK_TEMPLATE.format(
            system_persona=SYSTEM_PERSONA,
            few_shot_examples=FEW_SHOT_EXAMPLES,
            intent=intent,
            key_facts=key_facts,
            tone=tone,
        ),
        agent=writer,
        context=[plan_task],
        expected_output=(
            "A complete professional email with Subject line, salutation, "
            "body paragraphs, professional closing, and sign-off. "
            "No commentary — email only."
        ),
    )

    # Task 3: Quality critique (receives write_task context)
    critic_task = Task(
        description=CRITIC_TASK_TEMPLATE.format(
            intent=intent, key_facts=key_facts, tone=tone
        ),
        agent=critic,
        context=[write_task],
        expected_output=(
            "A structured four-point rubric evaluation covering: factual "
            "completeness, tone precision, structural integrity, and reader "
            "experience — with exactly four specific, quote-level improvements."
        ),
    )

    # Task 4: Final refinement (receives write + critic context)
    refine_task = Task(
        description=REFINER_TASK_TEMPLATE.format(
            system_persona=SYSTEM_PERSONA,
            few_shot_examples=FEW_SHOT_EXAMPLES,
            intent=intent,
            key_facts=key_facts,
            tone=tone,
        ),
        agent=refiner,
        context=[write_task, critic_task],
        expected_output=(
            "The final polished, send-ready email. "
            "Complete: Subject, salutation, body, closing, sign-off. "
            "All critique points addressed. No commentary."
        ),
    )

    crew = Crew(
        agents=[planner, writer, critic, refiner],
        tasks=[plan_task, write_task, critic_task, refine_task],
        process=Process.sequential,
        verbose=False,
    )

    # Retry loop with exponential backoff
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = crew.kickoff()
            raw    = str(result)
            email  = clean_email_output(raw)
            time.sleep(INTER_SCENARIO_WAIT)
            return email
        except Exception as exc:
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"    [retry {attempt}/{MAX_RETRIES}] {exc!r} — waiting {delay}s")
            time.sleep(delay)

    return f"[ERROR] All {MAX_RETRIES} retries failed for scenario {scenario['id']}."


# ── Batch generation ──────────────────────────────────────────────────────────

def generate_all(model_display_name: str, llm_id: str, scenarios: list) -> dict:
    """
    Generate emails for all scenarios using a single model.

    Parameters
    ----------
    model_display_name : str  — used only for console output
    llm_id             : str  — LiteLLM model string
    scenarios          : list — from scenarios.SCENARIOS

    Returns
    -------
    dict  {scenario_id (int) : generated_email (str)}
    """
    print(f"\n{'═'*60}")
    print(f"  GENERATION  |  {model_display_name}")
    print(f"{'═'*60}")

    planner, writer, critic, refiner = build_agents(llm_id)

    emails = {}
    for sc in scenarios:
        sid = sc["id"]
        intent_short = sc["intent"][:55]
        print(f"  [{sid:02d}/10] {intent_short}…", flush=True)
        email = generate_email(planner, writer, critic, refiner, sc)
        emails[sid] = email
        status = "✓" if not email.startswith("[ERROR]") else "✗"
        print(f"         {status}  {len(email)} chars generated.")

    return emails