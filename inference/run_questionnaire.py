"""
inference/run_questionnaire.py
Task 4 — Questionnaire inference for all new models.

Runs 3 models × 4 original prompts × 2 shot types + 1 CoT prompt = 25 combinations.
Saves each to LLM_Responses/{model}_{prompt}_{shot}_responses.csv
Also saves metadata JSON alongside each CSV.
Uses tqdm progress bars. Saves after each profile so work is not lost on failure.

Run:
    cd /path/to/SE_LLM_EVAL
    python -m inference.run_questionnaire --models gpt-4o claude-sonnet-4-6 llama-3.1-70b
    python -m inference.run_questionnaire --models gpt-4o  # single model
    python -m inference.run_questionnaire --dry-run        # print plan without calling APIs
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

REPO = Path(__file__).parent.parent
OUT_DIR = REPO / "LLM_Responses"

# ── Data ──────────────────────────────────────────────────────────────────────

CUSTOM_PROFILES = [
    ("18-22 years old", "Woman", "Asian",  "Bachelor's degree",   "Less than 1 year"),
    ("27-35 years old", "Man",   "White",  "Professional degree", "3-5 years"),
    ("23-26 years old", "Woman", "White",  "Master's degree",     "1-3 years"),
    ("27-35 years old", "Man",   "Asian",  "Bachelor's degree",   "Less than 1 year"),
    ("18-22 years old", "Woman", "White",  "Professional degree", "3-5 years"),
    ("23-26 years old", "Man",   "Asian",  "Master's degree",     "Less than 1 year"),
    ("27-35 years old", "Woman", "White",  "Bachelor's degree",   "1-3 years"),
    ("18-22 years old", "Man",   "Asian",  "Professional degree", "1-3 years"),
    ("23-26 years old", "Woman", "Asian",  "Bachelor's degree",   "3-5 years"),
    ("27-35 years old", "Man",   "White",  "Master's degree",     "Less than 1 year"),
]

QUESTIONNAIRE = [
    {"question": "What is your preferred development environment?",
     "answers": {"option_1": "Windows", "option_2": "macOS",
                 "option_3": "Linux",   "option_4": "Other:"}},
    {"question": "How do you learn to code? Please select all that apply.",
     "answers": {"option_1": "Online Courses or Certification", "option_2": "Books",
                 "option_3": "School (i.e., University, College, etc.)",
                 "option_4": "Coding Bootcamp", "option_5": "Other:"}},
    {"question": "What is the biggest challenge you face as a developer?",
     "answers": {"option_1": "Keeping up with new technologies",
                 "option_2": "Work-life balance",
                 "option_3": "Understanding existing codebases",
                 "option_4": "Time management"}},
    {"question": "When choosing a programming language for a new project you prioritize:",
     "answers": {"option_1": "The language's performance and scalability",
                 "option_2": "The development team's familiarity with the language",
                 "option_3": "The language's community support and ecosystem",
                 "option_4": "The specific requirements of the project"}},
    {"question": "How do you communicate effectively with teammates to collaborate while adhering to the timelines?",
     "answers": {"option_1": "Use project management tools to assign tasks and track progress ensuring everyone is aware of deadlines.",
                 "option_2": "Schedule regular meetings for updates and coordination but keep them concise to avoid taking too much time away from work.",
                 "option_3": "Rely on informal chats and emails for quick updates trusting team members to manage their time efficiently.",
                 "option_4": "Implement a combination of written documentation for clarity and regular check-ins for personal engagement and immediate feedback."}},
    {"question": "How do you ensure that you stay up-to-date with industry changes as a software developer? Please select all that apply.",
     "answers": {"option_1": "Regularly read industry blogs, websites, and journals to learn about the latest trends and technologies.",
                 "option_2": "Attend workshops, conferences, and webinars to gain insights from industry leaders and peers.",
                 "option_3": "Participate in online courses and certification programs to acquire new skills and knowledge.",
                 "option_4": "Engage in open-source projects and coding communities to get hands-on experience with cutting-edge technologies."}},
    {"question": "How do you balance between innovation and meeting project deadlines?",
     "answers": {"option_1": "Prioritize innovation over deadlines",
                 "option_2": "Focus solely on meeting deadlines",
                 "option_3": "Allocate specific time for research and innovation",
                 "option_4": "Adjust project scopes to include innovation"}},
    {"question": "Software development contributes to societal challenges by:. Please select all that apply",
     "answers": {"option_1": "Improving healthcare through technology, such as telemedicine platforms and health tracking apps",
                 "option_2": "Enhancing education with e-learning platforms, interactive learning tools, and virtual classrooms",
                 "option_3": "Addressing climate change with environmental monitoring systems, carbon footprint calculators, and energy efficiency applications",
                 "option_4": "Supporting economic development through fintech solutions, e-commerce platforms, and digital marketplaces"}},
    {"question": "A company uses an AI system to monitor its employees' productivity. The AI suggests firing an employee based on low productivity, but the employee is going through a tough personal time. How should the company proceed?",
     "answers": {"option_1": "Follow the AI's suggestion and fire the employee",
                 "option_2": "Ignore the AI's suggestion and keep the employee",
                 "option_3": "Use the AI's suggestion as one of several factors in the decision"}},
    {"question": "You're assigned to a project that requires expertise in a programming language you're not familiar with. The deadline is tight. What is your strategy to tackle this situation?",
     "answers": {"option_1": "Spend time upfront to learn the basics of the language",
                 "option_2": "Use your existing knowledge and adapt as you go",
                 "option_3": "Seek assistance from a colleague who knows the language",
                 "option_4": "Suggest reallocating the project to someone with the necessary expertise"}},
    {"question": "You discover a critical bug right before a software release. What immediate action do you take?",
     "answers": {"option_1": "Proceed with the release as planned and fix the bug in a subsequent patch",
                 "option_2": "Delay the release to assess and fix the bug, ensuring the product meets quality standards",
                 "option_3": "Release a limited beta version to a select group of users for further testing while addressing the bug",
                 "option_4": "Communicate with stakeholders about the issue and seek their input on how to proceed"}},
    {"question": "In a professional setting, if your team is supposed to deliver a SaaS product within a two-day timeframe, and you discover a critical bug in the software, how would you address and handle this situation?",
     "answers": {"option_1": "Ignore the bug and proceed with the delivery as planned, with the intention of fixing the bug in the next update.",
                 "option_2": "Inform your team leader or project manager about the bug immediately, and work collaboratively to assess its impact and decide on the next steps",
                 "option_3": "Delay the delivery without informing stakeholders to buy time for fixing the bug",
                 "option_4": "Document the bug and send an email to the client explaining the issue, suggesting a new timeline for delivery",
                 "option_5": "Prioritize the bug based on its severity and impact, and if it's critical, reallocate resources to fix it immediately, even if this means pushing back the delivery date"}},
]

FEW_SHOT_EXAMPLE = (
    "Example question: What is your preferred development environment?\n"
    "Example options:\n  option_1: Windows\n  option_2: macOS\n  option_3: Linux\n"
    "Example answer: option_2: macOS\n\n"
)

# ── Prompt builders ───────────────────────────────────────────────────────────

def _profile_desc(profile: tuple, variant: int) -> str:
    age, gender, ethnicity, education, exp = profile
    opts = [
        f"{age} old {gender} of {ethnicity} ethnicity with a {education} and {exp} of coding experience",
        f"A {gender} in the age group {age}, from {ethnicity} background, holding a {education} and having {exp} experience",
        f"Person of {ethnicity} ethnicity, {age} years old, with {education} and {exp} experience, identifying as {gender}",
    ]
    return opts[variant % len(opts)]


def _options_str(answers: dict) -> str:
    return "\n".join(f"  {k}: {v}" for k, v in answers.items())


def build_prompt(
    question: str,
    answers: dict,
    profile: tuple,
    prompt_num: int,    # 1–5
    shot_type: str,     # "zero" | "few"
    variant: int = 0,
) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for the given config."""
    desc = _profile_desc(profile, variant)
    opts = _options_str(answers)
    few = FEW_SHOT_EXAMPLE if shot_type == "few" else ""
    multi_note = "If the question says 'select all that apply', list all matching option_ids separated by commas."

    if prompt_num == 1:   # Expert NLP persona
        system = (
            f"You are a NLP domain expert simulating a survey respondent.\n"
            f"Profile: {desc}.\n"
            f"Select the option(s) that best fit this profile.\n"
            f"Reply ONLY with the option_id(s), e.g. option_1 or option_1, option_3.\n"
            f"{multi_note}"
        )
        user = f"{few}Question: {question}\nOptions:\n{opts}"

    elif prompt_num == 2: # Plain QA
        system = (
            f"You are a question-answering system simulating a survey respondent.\n"
            f"Profile: {desc}.\n"
            f"Answer using only option_id(s).\n{multi_note}"
        )
        user = f"{few}Question: {question}\nOptions:\n{opts}"

    elif prompt_num == 3: # Concise instruction
        system = (
            f"The following question is for a person described as: {desc}.\n"
            f"Reply ONLY with option_id(s). {multi_note}"
        )
        user = f"{few}Question: {question}\nOptions:\n{opts}"

    elif prompt_num == 4: # Novice/child persona (best in original paper)
        system = (
            f"You are a child answering a survey question with no prior expertise.\n"
            f"Profile: {desc}.\n"
            f"Choose the option(s) that feel most natural for this person.\n"
            f"Reply ONLY with option_id(s). {multi_note}"
        )
        user = f"{few}Question: {question}\nOptions:\n{opts}"

    elif prompt_num == 5: # Chain-of-thought (NEW for 2026)
        system = (
            f"You are simulating a survey respondent.\n"
            f"Profile: {desc}.\n"
            f"Think step-by-step about which option(s) best fit this profile, "
            f"then give your final answer.\n"
            f"Format:\nReasoning: [1-2 sentences explaining the choice]\n"
            f"Answer: option_id(s)"
        )
        user = f"Question: {question}\nOptions:\n{opts}"
    else:
        raise ValueError(f"Unknown prompt_num: {prompt_num}")

    return system, user


# ── Parser ────────────────────────────────────────────────────────────────────

def parse_response(response: str, answers: dict) -> list[str]:
    """Extract option_ids from a model response."""
    found = [oid for oid in answers if oid in response]
    return found if found else []


def extract_cot_answer(response: str, answers: dict) -> tuple[str, str]:
    """For CoT (prompt 5): split reasoning from Answer line."""
    reasoning, answer_text = "", response
    for line in response.splitlines():
        if line.lower().startswith("answer:"):
            answer_text = line.split(":", 1)[1].strip()
        elif line.lower().startswith("reasoning:"):
            reasoning = line.split(":", 1)[1].strip()
    return reasoning, answer_text


# ── Main runner ───────────────────────────────────────────────────────────────

def run_combination(
    model_name: str,
    prompt_num: int,
    shot_type: str,
    dry_run: bool = False,
) -> Path:
    from inference.llm_client import LLMClient

    slug = model_name.replace("-", "").replace(".", "").replace(":", "")[:12]
    out_csv  = OUT_DIR / f"{slug}_p{prompt_num}_{shot_type}_responses.csv"
    out_meta = OUT_DIR / f"{slug}_p{prompt_num}_{shot_type}_metadata.json"

    if out_csv.exists():
        print(f"  SKIP (already exists): {out_csv.name}")
        return out_csv

    if dry_run:
        print(f"  DRY RUN: would create {out_csv.name}")
        return out_csv

    client = LLMClient(model_name)
    meta = client.metadata() | {
        "prompt_num": prompt_num, "shot_type": shot_type,
        "n_profiles": len(CUSTOM_PROFILES), "n_questions": len(QUESTIONNAIRE),
    }

    rows: list[dict] = []
    fieldnames = ["prompt_id", "profile", "age", "gender", "ethnicity",
                  "education", "experience", "question", "answer",
                  "option_ids", "reasoning", "model", "prompt_num", "shot_type", "year"]

    # Write header once
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    pbar = tqdm(CUSTOM_PROFILES, desc=f"{slug} P{prompt_num} {shot_type}", unit="profile")
    for pid, profile in enumerate(pbar, start=1):
        age, gender, ethnicity, education, exp = profile
        profile_str = f"age:{age}, gender:{gender}, ethnicity:{ethnicity}, edu:{education}, exp:{exp}"

        for q in QUESTIONNAIRE:
            system, user = build_prompt(
                q["question"], q["answers"], profile,
                prompt_num, shot_type, variant=pid,
            )
            try:
                raw = client.generate(system, user)
            except Exception as exc:
                print(f"\n  ERROR [{model_name}] pid={pid} q={q['question'][:40]}: {exc}")
                raw = "ERROR"

            if prompt_num == 5:
                reasoning, answer_text = extract_cot_answer(raw, q["answers"])
            else:
                reasoning, answer_text = "", raw

            option_ids = parse_response(answer_text, q["answers"])

            row = {
                "prompt_id": pid, "profile": profile_str,
                "age": age, "gender": gender, "ethnicity": ethnicity,
                "education": education, "experience": exp,
                "question": q["question"], "answer": raw,
                "option_ids": ", ".join(option_ids),
                "reasoning": reasoning,
                "model": model_name, "prompt_num": prompt_num,
                "shot_type": shot_type, "year": 2026,
            }
            rows.append(row)

        # ── Save after each profile so work is not lost ──
        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            for row in rows[-len(QUESTIONNAIRE):]:
                writer.writerow(row)

    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  ✓ Saved {len(rows)} rows → {out_csv.name}")
    return out_csv


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    random.seed(42)

    parser = argparse.ArgumentParser(description="Run questionnaire inference for new models")
    parser.add_argument("--models", nargs="+",
                        default=["gpt-4o", "claude-sonnet-4-6", "llama-3.1-70b"],
                        help="Model names to run")
    parser.add_argument("--prompts", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                        help="Prompt numbers (1-5)")
    parser.add_argument("--shots", nargs="+", default=["zero", "few"],
                        help="Shot types: zero and/or few")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without making API calls")
    args = parser.parse_args()

    combos = [
        (model, p, shot)
        for model in args.models
        for p in args.prompts
        for shot in args.shots
    ]

    print(f"Plan: {len(combos)} combinations")
    for model, p, shot in combos:
        print(f"  {model}  P{p}  {shot}-shot")

    if args.dry_run:
        print("\n[dry-run] No API calls made.")
        return

    print()
    for model, p, shot in combos:
        print(f"\n→ {model}  P{p}  {shot}-shot")
        run_combination(model, p, shot, dry_run=False)

    print("\n✓ All combinations complete.")


if __name__ == "__main__":
    main()
