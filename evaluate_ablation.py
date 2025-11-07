# evaluator for A/B (Baseline vs Sentiment-Aware) chatbot output

import argparse
import pandas as pd
import csv
import re
import sys
import math
from collections import defaultdict
from sentiment_analysis.src.sentiment_api import SentimentAnalyzer
from rag.retriever import Retriever
from llm.clean import format_context_for_llm

# --- TEXT PROCESSING HELPERS ---
_word_re = re.compile(r"[A-Za-z][A-Za-z'\-]+")
_sentence_split_re = re.compile(r"[.!?]+[\s\)]|[\n]+")

def tokenize(text: str) -> list:
    ''' Simple whitespace tokenizer '''
    return _word_re.findall(text or "")

def sentences(text: str) -> list:
    ''' Simple sentence splitter '''
    splits = _sentence_split_re.split(text or "")
    return [s.strip() for s in splits if s.strip()]

def ngrams(tokens, n):
    return list(zip(*[tokens[i:] for i in range(n)]))

def jaccard_bigram(a: str, b: str) -> float:
    ta, tb = tokenize(a.lower()), tokenize(b.lower())
    if len(ta) < 2 or len(tb) < 2:
        return 0.0
    A, B = set(ngrams(ta, 2)), set(ngrams(tb, 2))
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def contains_validation_phrase(text: str) -> bool:
    ''' Check if text contains phrases indicating validation '''
    validation_phrases = [
        "i understand",
        "that sounds",
        "i'm sorry to hear",
        "i can see how",
        "it can be overwhelming",
        "i get that",
        "i know how",
        "it must be tough",
    ]
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in validation_phrases)


def indecision_in_query(query: str) -> bool:
    ''' Check if user query indicates indecision '''
    indecision_phrases = [
        "i don't know",
        "can't decide",
        "not sure",
        "what do you suggest",
        "help me choose",
        "i'm undecided",
        "any recommendations",
    ]
    query_lower = query.lower()
    return any(phrase in query_lower for phrase in indecision_phrases)

def proper_name_candidates(text: str) -> list:
    ''' Extract proper names from context for checking recommendations '''
    toks = re.findall(r"[A-Z][a-zA-Z'&.-]+", text or "")
    return set(toks)

# --- CONTEXT PARSING AND VENUE EXTRACTION ---
def extract_context_venues_and_snippets(context_items):
    ''' Extract venue names and their associated snippets from context '''
    venue_names = set()
    name_map = {}
    snippets_by_name = defaultdict(list)

    for it in context_items:
        name = (it.get("name") or "Unknown Restaurant").strip()
        lname = name.lower()
        if lname not in name_map:
            name_map[lname] = name
        venue_names.add(lname)
        snip = (it.get("chunk_text") or "").strip()
        if snip:
            snippets_by_name[lname].append(snip)

    return venue_names, name_map, snippets_by_name


def venues_mentioned(response_text: str, name_map):
    """Return set of lowercased venue names that are explicitly mentioned in response by substring."""
    rlow = (response_text or "").lower()
    mentioned = set()
    for lname, true_name in name_map.items():
        # direct substring match; can be improved with fuzzy matching
        if true_name.lower() in rlow:
            mentioned.add(lname)
    return mentioned

# --- POLICUY AND OVERLOAD CHECKS ---
def policy_compliance(expected_sent: str, query: str, response: str, venue_count: int) -> bool:
    s = (expected_sent or "").lower()
    resp = response or ""

    if s == "positive":
        return 2 <= venue_count <= 3

    if s == "neutral":
        if indecision_in_query(query):
            # expect one short clarifying question and not flooding with options
            has_q = resp.count("?") >= 1
            return has_q and venue_count <= 1
        else:
            return 1 <= venue_count <= 2

    if s == "negative":
        return venue_count <= 2 and contains_validation_phrase(resp)

    # fallback
    return True


def overload_flag(expected_sent: str, sentence_count: int, venue_count: int) -> bool:
    s = (expected_sent or "").lower()
    if s == "positive":
        return sentence_count > 7 or venue_count > 4
    else:
        return sentence_count > 5 or venue_count > 3
    

# --- MAIN EVALUATION PIPELINE ---
def evaluate(eval_csv: str, output_csv: str, top_k: int = 5):
    # Init models
    print("[Init] Loading SentimentAnalyzer and Retriever ...")
    analyzer = SentimentAnalyzer()
    retriever = Retriever()

    # Load eval CSV
    if pd:
        df = pd.read_csv(eval_csv)
        records = df.to_dict(orient="records")
    else:
        records = []
        with open(eval_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)

    rows_out = []
    agg = defaultdict(list)  # key: (variant, expected_sentiment) -> list of dict metrics

    print(f"[Run] Evaluating {len(records)} rows ...")

    for r in records:
        query = r.get("Query", "")
        expected_sent = (r.get("Expected Sentiment") or "").strip().lower()

        # recompute context to know the allowed venues/snippets
        ctx_items = retriever.retrieve(query, top_k=top_k, analyzer=analyzer)
        venue_names, name_map, snippets_by_name = extract_context_venues_and_snippets(ctx_items)

        # Responses
        respA = r.get("Response_A_Baseline", "") or ""
        respB = r.get("Response_B_Sentiment_Aware", "") or ""

        for variant, text in (("A", respA), ("B", respB)):
            toks = tokenize(text)
            sents = sentences(text)
            sent_count = len(sents)
            tok_count = len(toks)

            mentioned = venues_mentioned(text, name_map)
            venue_count = len(mentioned)

            # Faithfulness / hallucination:
            # grounded_any: mentions at least one allowed venue
            grounded_any = venue_count > 0
            # OOV proper name heuristic: any TitleCase tokens not part of allowed venues (very rough)
            proper_names = proper_name_candidates(text)
            allowed_name_tokens = set()
            for nm in name_map.values():
                allowed_name_tokens.update(nm.split())
            oov_candidates = [p for p in proper_names if p not in allowed_name_tokens]
            hallucination = (len(oov_candidates) > 0 and not grounded_any)

            # Snippet grounding: compare to concatenated snippets of mentioned venues (or all if none)
            if mentioned:
                ref_text = "\n".join(sum([snippets_by_name[l] for l in mentioned], []))
            else:
                # fallback: all snippets of top_k venues
                ref_text = "\n".join(sum(snippets_by_name.values(), []))
            grounding = jaccard_bigram(text, ref_text)

            # Sentiment adherence: label the response itself
            try:
                lab = analyzer.analyze(text).get("sentiment", "").lower()
            except Exception:
                lab = ""
            sent_adherence = 1 if (lab == expected_sent) else 0

            # Policy compliance & overload
            policy_ok = policy_compliance(expected_sent, query, text, venue_count)
            overload = overload_flag(expected_sent, sent_count, venue_count)

            row = {
                "Query": query,
                "ExpectedSentiment": expected_sent,
                "Variant": variant,
                "Tokens": tok_count,
                "Sentences": sent_count,
                "VenuesMentioned": venue_count,
                "GroundedAny": int(grounded_any),
                "Hallucination": int(hallucination),
                "GroundingScoreBigramJ": round(grounding, 4),
                "SentimentAdherence": int(sent_adherence),
                "PolicyCompliance": int(policy_ok),
                "Overload": int(overload),
            }
            rows_out.append(row)
            agg[(variant, expected_sent)].append(row)
            agg[(variant, "ALL")].append(row)

    # Write detailed CSV
    fieldnames = list(rows_out[0].keys()) if rows_out else []
    if output_csv:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in rows_out:
                w.writerow(row)
        print(f"[OK] Wrote detailed metrics → {output_csv}")

    # Print aggregates
    def summarize(key):
        items = agg.get(key, [])
        if not items:
            return None
        n = len(items)
        mean = lambda k: sum(x[k] for x in items) / n if n else 0.0
        pct = lambda k: 100.0 * sum(x[k] for x in items) / n if n else 0.0
        return {
            "n": n,
            "Tokens": round(mean("Tokens"), 1),
            "Sentences": round(mean("Sentences"), 2),
            "Venues": round(mean("VenuesMentioned"), 2),
            "GroundedAny%": round(pct("GroundedAny"), 1),
            "Hallucination%": round(pct("Hallucination"), 1),
            "GroundingScore": round(mean("GroundingScoreBigramJ"), 3),
            "SentimentAdherence%": round(pct("SentimentAdherence"), 1),
            "PolicyCompliance%": round(pct("PolicyCompliance"), 1),
            "Overload%": round(pct("Overload"), 1),
        }

    sentiments = ["ALL", "positive", "neutral", "negative"]
    print("\n=== Aggregate Summary ===")
    header = ["Variant","Sentiment","n","Tokens","Sentences","Venues","GroundedAny%","Hallucination%","GroundingScore","SentimentAdherence%","PolicyCompliance%","Overload%"]
    print("\t".join(header))
    for variant in ["A","B"]:
        for s in sentiments:
            sm = summarize((variant, s))
            if sm:
                print("\t".join(map(str, [
                    variant, s, sm["n"], sm["Tokens"], sm["Sentences"], sm["Venues"],
                    sm["GroundedAny%"], sm["Hallucination%"], sm["GroundingScore"],
                    sm["SentimentAdherence%"], sm["PolicyCompliance%"], sm["Overload%"]
                ])))

    print("\n[Done] Use the table above in Results, and include eval_metrics_detailed.csv as a supplemental artifact.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_csv", type=str, default="evaluation_results.csv", help="CSV produced by run_eval.py")
    ap.add_argument("--output_csv", type=str, default="eval_metrics_detailed.csv", help="Where to write per-query metrics")
    ap.add_argument("--top_k", type=int, default=5, help="top_k for retriever to rebuild Context")
    args = ap.parse_args()
    evaluate(args.eval_csv, args.output_csv, top_k=args.top_k)