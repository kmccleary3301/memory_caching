from __future__ import annotations

import re
import string


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def normalize_text(text: str) -> str:
    text = text.lower()
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    text = _ARTICLES_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def exact_match(prediction: str, answer: str) -> float:
    return 1.0 if normalize_text(prediction) == normalize_text(answer) else 0.0


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(normalize_text(text))


def token_f1(prediction: str, answer: str) -> float:
    pred_tokens = _tokens(prediction)
    ans_tokens = _tokens(answer)
    if len(pred_tokens) == 0 or len(ans_tokens) == 0:
        return 0.0
    pred_counts: dict[str, int] = {}
    ans_counts: dict[str, int] = {}
    for tok in pred_tokens:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1
    for tok in ans_tokens:
        ans_counts[tok] = ans_counts.get(tok, 0) + 1

    overlap = 0
    for tok in pred_counts:
        overlap += min(pred_counts.get(tok, 0), ans_counts.get(tok, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ans_tokens)
    return float(2 * precision * recall / (precision + recall))


def rouge_l_f1(prediction: str, answer: str) -> float:
    pred_tokens = _tokens(prediction)
    ans_tokens = _tokens(answer)
    if len(pred_tokens) == 0 or len(ans_tokens) == 0:
        return 0.0

    dp = [[0] * (len(ans_tokens) + 1) for _ in range(len(pred_tokens) + 1)]
    for i in range(1, len(pred_tokens) + 1):
        for j in range(1, len(ans_tokens) + 1):
            if pred_tokens[i - 1] == ans_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ans_tokens)
    return float(2 * precision * recall / (precision + recall))


def extract_answer_candidates(prediction: str) -> list[str]:
    cleaned = prediction.replace(",", " ").replace(";", " ").replace("|", " ")
    tokens = [tok.strip() for tok in cleaned.split() if tok.strip()]
    if len(tokens) == 0:
        return []
    return tokens
