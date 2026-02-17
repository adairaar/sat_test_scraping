"""
sat_extractor.py
================
Extracts SAT practice test questions and correct answers from two PDFs:
  1. The test questions PDF  (e.g. sat-practice-test-8-digital.pdf)
  2. The scoring guide PDF   (e.g. scoring-sat-practice-test-8-digital.pdf)

Produces a single DataFrame / CSV with columns:
  test_name, section, module, question_number, question_type, correct_answer,
  passage, question_stem, choice_A, choice_B, choice_C, choice_D

Usage (as a script)
-------------------
Edit the three path constants at the bottom of this file, then run:
    python sat_extractor.py

Usage (as a module)
-------------------
    from sat_extractor import main
    df = main(
        questions_pdf_path = "sat-practice-test-8-digital.pdf",
        scoring_pdf_path   = "scoring-sat-practice-test-8-digital.pdf",
        output_csv_path    = "sat_questions.csv",   # optional
    )

Requires:  pip install pdfplumber pandas
"""

import re
from collections import defaultdict

import pandas as pd
import numpy as np
import glob
import pdfplumber
import sqlite3
from sqlalchemy import create_engine, text, Index


# ═════════════════════════════════════════════════════════════════════════════
# PART 1 – Extract questions from the test PDF
# ═════════════════════════════════════════════════════════════════════════════

# ── Section / page detection ──────────────────────────────────────────────────
RW_HEADER  = re.compile(r"Reading and Writ\w*\s+33 QUESTIONS", re.I | re.S)
MATH_HEADER = re.compile(r"Math\s+27 QUESTIONS", re.I | re.S)
SKIP_RE = re.compile(
    r"No Test Material|"
    r"For multiple-choice questions,\s+solve each|"
    r"For student-produced response questions,\s+solve",
    re.I,
)

# ── Per-line noise ────────────────────────────────────────────────────────────
Q_NOISE_RE = re.compile(
    r"Unauthorized copying|reuse of any part|"
    r"^(CONTINUE|STOP|STO)$|"
    r"^-+(\s*-+)*$|^\.{10,}$|"
    r"^Modu(le)?$|^ule$|^OP$|"
    r"^Reading and Writ|"
    r"^33 QUESTIONS$|^27 QUESTIONS$|"
    r"^Math$|^©|^The SAT|"
    r"^check your work|^er module|^y check your work",
    re.I,
)

BARE_INT_RE     = re.compile(r"^\d{1,3}$")
CHOICE_START_RE = re.compile(r"^([A-D])\)\s*(.*)$")

STEM_PATS = [re.compile(p, re.I) for p in [
    r"Which choice",
    r"Which of the following",
    r"According to the (text|passage)",
    r"Based on the (text|table|information|graph|figure|data|passage)",
    r"What (is|are|does|was|were|value)",
    r"How (did|does|many|much|would)",
    r"The (student|researcher|author|narrator|producer) (want|claim|argue|suggest|is)",
    r"If .{3,60}(true|correct|value)",
    r"Which (statement|equation|expression|finding|point|of the following)",
    r"For (what|which) value",
    r"In the (xy-plane|given|figure)",
    r"The (given|following) (equation|system|expression)",
    r"What is the (value|equation|measure|y-intercept|area|probability)",
    r"Which equation",
    r"What was",
]]


def _clean(s):
    return re.sub(r"\s{2,}", " ", s.strip())


def _is_noise(s):
    return bool(Q_NOISE_RE.search(s))


def _split_passage_stem(lines):
    idx = None
    for i, ln in enumerate(lines):
        for p in STEM_PATS:
            if p.search(ln):
                idx = i
                break
    if idx is None:
        return "", " ".join(lines).strip()
    return " ".join(lines[:idx]).strip(), " ".join(lines[idx:]).strip()


def _page_lines(page):
    """Return lines from left column then right column."""
    w, h = page.width, page.height
    lines = []
    for x0, x1 in [(0, w * 0.5), (w * 0.5, w)]:
        txt = page.crop((x0, 0, x1, h)).extract_text(x_tolerance=3, y_tolerance=3) or ""
        for raw in txt.split("\n"):
            c = _clean(raw)
            if c:
                lines.append(c)
    return lines


class _Tracker:
    def __init__(self):
        self.section = "Unknown"
        self.module  = "1"
        self._rw     = 0
        self._math   = 0

    def update(self, full_text):
        if RW_HEADER.search(full_text):
            self._rw    += 1
            self.section = "Reading and Writing"
            self.module  = str(self._rw)
        elif MATH_HEADER.search(full_text):
            self._math  += 1
            self.section = "Math"
            self.module  = str(self._math)

    @property
    def max_q(self):
        return 33 if "Reading" in self.section else 27


def extract_questions(questions_pdf_path):
    """
    Parse the SAT test PDF and return a list of question dicts.

    Parameters
    ----------
    questions_pdf_path : str or Path

    Returns
    -------
    list of dict with keys:
        test_name, section, module, question_number, question_type,
        passage, question_stem, choice_A, choice_B, choice_C, choice_D
    """
    tracker = _Tracker()
    records = {}

    cur_qnum         = None
    cur_lines        = []
    cur_section      = "Unknown"
    cur_module       = "1"
    cur_choices      = {}
    cur_choice_letter = None

    def flush_choice():
        nonlocal cur_choice_letter
        if cur_choice_letter and cur_choice_letter in cur_choices:
            cur_choices[cur_choice_letter] = " ".join(cur_choices[cur_choice_letter]).strip()
        cur_choice_letter = None

    def flush_q():
        nonlocal cur_qnum, cur_lines, cur_choices, cur_section, cur_module, cur_choice_letter
        flush_choice()
        if cur_qnum is None:
            return
        passage, stem = _split_passage_stem(cur_lines)
        q_type = "Multiple Choice" if len(cur_choices) >= 2 else "Free Response"
        key    = f"{cur_section}|{cur_module}|{cur_qnum}"
        new_len = len(passage + stem)
        if key in records:
            old = records[key]
            if new_len <= len(old.get("passage", "") + old.get("question_stem", "")):
                cur_qnum = None; cur_lines = []; cur_choices = {}; cur_choice_letter = None
                return
        records[key] = dict(
            test_name       = questions_pdf_path.split('/')[-1].split('.')[0], #"SAT Practice Test #8",
            section         = cur_section,
            module          = cur_module,
            question_number = cur_qnum,
            question_type   = q_type,
            passage         = passage,
            question_stem   = stem,
            choice_A        = cur_choices.get("A", ""),
            choice_B        = cur_choices.get("B", ""),
            choice_C        = cur_choices.get("C", ""),
            choice_D        = cur_choices.get("D", ""),
        )
        cur_qnum = None; cur_lines = []; cur_choices = {}; cur_choice_letter = None

    with pdfplumber.open(str(questions_pdf_path)) as pdf:
        for page in pdf.pages:
            full = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            tracker.update(full)
            if SKIP_RE.search(full):
                flush_q()
                continue

            prev_was_q = False
            for line in _page_lines(page):
                if _is_noise(line):
                    prev_was_q = False
                    continue

                if BARE_INT_RE.match(line):
                    val = int(line)
                    if 1 <= val <= tracker.max_q and not prev_was_q:
                        flush_q()
                        cur_qnum         = val
                        cur_section      = tracker.section
                        cur_module       = tracker.module
                        cur_lines        = []
                        cur_choices      = {}
                        cur_choice_letter = None
                        prev_was_q       = True
                    else:
                        prev_was_q = False
                    continue

                prev_was_q = False
                if cur_qnum is None:
                    continue

                m = CHOICE_START_RE.match(line)
                if m:
                    flush_choice()
                    cur_choice_letter = m.group(1)
                    rest = m.group(2).strip()
                    cur_choices[cur_choice_letter] = [rest] if rest else []
                    continue

                if cur_choice_letter is not None:
                    if any(p.search(line) for p in STEM_PATS):
                        flush_choice()
                        cur_lines.append(line)
                    else:
                        cur_choices[cur_choice_letter].append(line)
                    continue

                cur_lines.append(line)

    flush_q()

    def sort_key(r):
        return (0 if "Reading" in r["section"] else 1, r["module"], r["question_number"])

    return sorted(records.values(), key=sort_key)


# ═════════════════════════════════════════════════════════════════════════════
# PART 2 – Extract answer key from the scoring guide PDF
# ═════════════════════════════════════════════════════════════════════════════

COLUMN_ORDER = [
    ("Reading and Writing", 1),
    ("Reading and Writing", 2),
    ("Math",                1),
    ("Math",                2),
]

AK_Q_NUM_RE = re.compile(r"^\d{1,2}$")

AK_NOISE_RE = re.compile(
    r"QUESTION|MODULE|Reading|Writing|Math|Correct|CORRECT|MARK|RAW|"
    r"SCORE|Total|NOITSEUQ|TCERROC|KRAM|RUOY|SREWSNA|Worksheet|"
    r"Answer|Key|Section|Module|SECTION|\+|=",
    re.IGNORECASE,
)

ANSWER_KEY_PAGE_RE = re.compile(r"Answer\s+Key", re.IGNORECASE)


def _count_answer_pairs(page):
    words = page.extract_words(x_tolerance=3, y_tolerance=3)
    rows_by_y = defaultdict(list)
    for w in words:
        rows_by_y[round(w["top"] / 2) * 2].append(w)
    count = 0
    for y in sorted(rows_by_y):
        line   = sorted(rows_by_y[y], key=lambda w: w["x0"])
        tokens = [w["text"] for w in line]
        if any(AK_NOISE_RE.search(t) for t in tokens):
            continue
        for i, t in enumerate(tokens):
            if (AK_Q_NUM_RE.match(t)
                    and i + 1 < len(tokens)
                    and not AK_Q_NUM_RE.match(tokens[i + 1])):
                count += 1
    return count


def _find_answer_key_page(pdf):
    candidates = []
    for page in pdf.pages:
        text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
        if ANSWER_KEY_PAGE_RE.search(text):
            candidates.append((page, _count_answer_pairs(page)))
    if not candidates:
        raise ValueError(
            "Could not find an 'Answer Key' page in the scoring PDF. "
            "Ensure you are using the official SAT scoring guide."
        )
    return max(candidates, key=lambda t: t[1])[0]


def _find_column_centres(page, n_cols=4):
    words = page.extract_words(x_tolerance=3, y_tolerance=3)
    rows_by_y = defaultdict(list)
    for w in words:
        rows_by_y[round(w["top"] / 2) * 2].append(w)

    q_xs = []
    for y in sorted(rows_by_y):
        line   = sorted(rows_by_y[y], key=lambda w: w["x0"])
        tokens = [(w["text"], w["x0"]) for w in line]
        if any(AK_NOISE_RE.search(t) for t, _ in tokens):
            continue
        for i, (text, x) in enumerate(tokens):
            if (AK_Q_NUM_RE.match(text)
                    and i + 1 < len(tokens)
                    and not AK_Q_NUM_RE.match(tokens[i + 1][0])):
                q_xs.append(x)

    if not q_xs:
        return []

    unique_xs = sorted(set(round(x) for x in q_xs))
    if len(unique_xs) <= n_cols:
        return [float(x) for x in unique_xs]

    gaps = sorted(
        [(unique_xs[i + 1] - unique_xs[i], i) for i in range(len(unique_xs) - 1)],
        reverse=True,
    )
    split_after = sorted(idx for _, idx in gaps[: n_cols - 1])

    groups, prev = [], 0
    for si in split_after:
        groups.append(unique_xs[prev: si + 1])
        prev = si + 1
    groups.append(unique_xs[prev:])

    return [sum(g) / len(g) for g in groups if g]


def _extract_answer_triples(page, centres):
    page_width = page.width

    right_limits = []
    for i, c in enumerate(centres):
        right_limits.append(
            (centres[i] + centres[i + 1]) / 2 if i + 1 < len(centres) else page_width
        )

    def col_of(x):
        return min(range(len(centres)), key=lambda c: abs(centres[c] - x))

    words = page.extract_words(x_tolerance=3, y_tolerance=3)
    rows_by_y = defaultdict(list)
    for w in words:
        rows_by_y[round(w["top"] / 2) * 2].append(w)

    triples = []

    for y in sorted(rows_by_y):
        line   = sorted(rows_by_y[y], key=lambda w: w["x0"])
        tokens = [(w["text"], w["x0"]) for w in line]
        if any(AK_NOISE_RE.search(t) for t, _ in tokens):
            continue

        i = 0
        while i < len(tokens):
            text, x = tokens[i]
            if not AK_Q_NUM_RE.match(text):
                i += 1
                continue

            ci          = col_of(x)
            right_limit = right_limits[ci]

            ans_parts = []
            j = i + 1
            while j < len(tokens):
                nxt_text, nxt_x = tokens[j]
                if nxt_x >= right_limit:
                    break
                if AK_Q_NUM_RE.match(nxt_text):
                    # Keep single-digit numeric answers (e.g. "9") unless
                    # followed by a non-digit in the same zone (new Q-A pair)
                    is_new_q = (
                        j + 1 < len(tokens)
                        and not AK_Q_NUM_RE.match(tokens[j + 1][0])
                        and tokens[j + 1][1] < right_limit
                    )
                    if is_new_q:
                        break
                    ans_parts.append(nxt_text)
                else:
                    ans_parts.append(nxt_text)
                j += 1

            if ans_parts:
                triples.append((int(text), x, " ".join(ans_parts).strip()))

            i = j if ans_parts else i + 1

    return triples


def extract_answer_key(scoring_pdf_path):
    """
    Parse the SAT scoring guide PDF and return the complete answer key.

    Parameters
    ----------
    scoring_pdf_path : str or Path

    Returns
    -------
    dict  { (section: str, module: int) : { question_number (int): answer (str) } }
    """
    with pdfplumber.open(str(scoring_pdf_path)) as pdf:
        ak_page = _find_answer_key_page(pdf)
        centres = _find_column_centres(ak_page, n_cols=len(COLUMN_ORDER))
        if not centres:
            raise ValueError("Could not detect answer-key columns in the scoring PDF.")
        triples = _extract_answer_triples(ak_page, centres)

    if not triples:
        raise ValueError("No answer pairs found in the scoring PDF.")

    def col_of(x):
        return min(range(len(centres)), key=lambda c: abs(centres[c] - x))

    raw_cols = defaultdict(dict)
    for qnum, x, answer in triples:
        ci = col_of(x)
        if qnum not in raw_cols[ci]:
            raw_cols[ci][qnum] = answer

    key = {}
    for col_idx, (section, module) in enumerate(COLUMN_ORDER):
        answers = raw_cols.get(col_idx, {})
        key[(section, module)] = answers
        if answers:
            print(f"  {section} Module {module}: "
                  f"{len(answers)} answers (Q{min(answers)}–Q{max(answers)})")
        else:
            print(f"  {section} Module {module}: 0 answers (column not detected)")

    return key


# ═════════════════════════════════════════════════════════════════════════════
# PART 3 – Merge and entry point
# ═════════════════════════════════════════════════════════════════════════════

def main(questions_pdf_path, scoring_pdf_path, output_csv_path=None):
    """
    Full pipeline: extract questions from the test PDF, extract the answer
    key from the scoring guide PDF, merge them, and optionally save to CSV.

    Parameters
    ----------
    questions_pdf_path : str or Path
        The SAT practice test PDF (e.g. sat-practice-test-8-digital.pdf).
    scoring_pdf_path : str or Path
        The College Board scoring guide PDF
        (e.g. scoring-sat-practice-test-8-digital.pdf).
    output_csv_path : str or Path, optional
        If provided, the merged DataFrame is saved here as a CSV.

    Returns
    -------
    pd.DataFrame
        One row per question with a correct_answer column.
    """
    # ── Step 1: extract questions ──────────────────────────────────
    print(f"\n[1] Extracting questions from:\n    {questions_pdf_path}")
    records = extract_questions(questions_pdf_path)
    print(f"    {len(records)} questions extracted.")

    df = pd.DataFrame(records)
    col_order = [
        "test_name", "section", "module", "question_number", "question_type",
        "passage", "question_stem",
        "choice_A", "choice_B", "choice_C", "choice_D",
    ]
    df = df[[c for c in col_order if c in df.columns]]

    # ── Step 2: clean up noise rows ────────────────────────────────
    df = df[df["section"] != "Unknown"].copy()
    GARBAGE = re.compile(r"DIRECTIONS|NOTES Unless otherwise|A =nr2|C=2nr", re.I)
    df = df[~df["question_stem"].str.contains(GARBAGE, na=False)].copy()
    df = df.reset_index(drop=True)

    # ── Step 3: extract answer key ─────────────────────────────────
    print(f"\n[2] Extracting answer key from:\n    {scoring_pdf_path}")
    key = extract_answer_key(scoring_pdf_path)

    # ── Step 4: merge correct_answer column ────────────────────────
    print("\n[3] Merging answers...")

    def lookup(row):
        try:
            mod  = int(row["module"])
            qnum = int(row["question_number"])
        except (ValueError, TypeError):
            return ""
        return key.get((row["section"], mod), {}).get(qnum, "")

    df["correct_answer"] = df.apply(lookup, axis=1)

    # Insert correct_answer right after question_type
    cols = list(df.columns)
    cols.remove("correct_answer")
    cols.insert(cols.index("question_type") + 1, "correct_answer")
    df = df[cols]

    matched   = (df["correct_answer"] != "").sum()
    unmatched = (df["correct_answer"] == "").sum()
    print(f"    Matched: {matched}  |  Unmatched: {unmatched}")

    # ── Step 5: save ───────────────────────────────────────────────
    if output_csv_path:
        df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
        print(f"\n[4] Saved → {output_csv_path}")

    return df


if __name__ == "__main__":
    # Get all SAT tests available in directory
    # Assumes each test has an answer key
    sat_dir = f"/sat_tests/"
    files_pdf = glob.glob(sat_dir + "*.pdf")

    questions_list = np.sort([f for f in files_pdf if f.startswith(sat_dir + "sat-practice-test")])
    scoring_answers_list = np.sort([f for f in files_pdf if f.startswith(sat_dir + "scoring-")])

    df_list = []

    for i in range(len(questions_list)):

        df_out = main(
            scoring_pdf_path   = scoring_answers_list[i],
            questions_pdf_path = questions_list[i],
            output_csv_path    = f"sat_questions_with_answers{i}.csv",
        )

        df_list.append(df_out)

    print("saving all results to csv...")

    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df['key'] = combined_df['test_name'] + "__" + \
                            combined_df['section'] + "__" + \
                            combined_df['module'].apply(lambda x: str(x)) + "__" + \
                            combined_df['question_number'].apply(lambda x: str(x))
    combined_df['key'] = combined_df['key'].apply(lambda x: x.replace(" ", "_"))
    combined_df.to_csv('all_df.csv', index=False)


    print("saving all results to SQL database...")
    engine = create_engine('sqlite:///outputs/sat_questions.db')
    # Write DataFrame to SQL table
    combined_df.to_sql("sat_questions", engine, if_exists="replace", index=False)

    print("Done!")
