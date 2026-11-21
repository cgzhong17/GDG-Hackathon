"""Extractive, precedent-grounded reply composer (fully local, no generative LLM).

Every sentence in the body of the draft is copied (or machine-translated) from the official
answer of a retrieved historical ticket and tagged with that ticket's id, so citations are
correct by construction and the system cannot invent policies, prices or guarantees.

Steps
  1. split the top-k precedent answers into sentences
  2. drop unusable sentences: placeholders (<name>, <tel_num> ...), greetings / sign-offs,
     and commitments that only applied to that other customer (money, %, deadlines,
     guarantees, refunds)
  3. score sentences by e5 similarity to the incoming ticket, weighted by precedent similarity
  4. greedily pick 2-4 non-redundant sentences, keep them in natural order
  5. translate sentences whose language differs from the ticket (opus-mt, local)
  6. wrap in a fixed EN/DE opening + closing; add a "specialist will follow up, no commitment"
     line when the customer asks for refunds / legal / pricing commitments
"""
import re
from functools import lru_cache

import numpy as np

from lang import detect_language

PLACEHOLDER = re.compile(r"<[^<>]{1,30}>|\[[^\[\]]{1,30}\]|\{[^{}]{1,30}\}")
GREETING = re.compile(
    r"^\s*(dear|hello|hi|good (morning|afternoon)|sehr geehrte|liebe[rs]?|hallo|guten tag)\b|"
    r"(best|kind|warm) regards|sincerely|mit freundlichen grüßen|beste grüße|viele grüße|"
    r"^\s*(thank you|thanks|vielen dank|danke)\b[^.]{0,40}[.!]?\s*$", re.I)
COMMITMENT = re.compile(
    r"[$€£]\s?\d|\d\s?(%|€|\$|eur|usd|euro|dollar|prozent|percent)|"
    r"\b\d+\s?(hours?|days?|weeks?|months?|minutes?|business days|stunden|tage[n]?|wochen|monate[n]?|minuten|werktage[n]?)\b|"
    r"\b(refund|reimburse|compensat|credit (note|to your)|guarantee|warrant|discount|free of charge|waive|"
    r"erstatt|rückerstatt|gutschrift|entschädig|garantie|rabatt|kostenlos|kostenfrei)", re.I)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ0-9\"(])|\n+")
# restates the *other* customer's issue -> useless / misleading for this customer (our OPEN line acknowledges)
ACK = re.compile(
    r"^(thank you|thanks|we (have )?received|we acknowledge|acknowledged|regarding|in response to|we appreciate|"
    r"we understand|we are aware|vielen dank|danke|wir haben (ihre|die|eine)|bezüglich|wir bestätigen|"
    r"wir verstehen|wir sind uns bewusst|in bezug auf|(ich |wir )?bestätige[n]?|(ich |wir )?verstehe[n]?|"
    r"ich habe|we noticed|i understand|i acknowledge|confirm(ing)? (that )?(we|i) (have )?received)", re.I)
# customer-voice or fragment sentences that leaked into the dataset answers
CUSTOMER_VOICE = re.compile(r"\b(i am writing to|i would like to (inform|report)|ich möchte (ihnen )?mitteilen|ich schreibe)\b"
                            r"|^customer\b|^kunde\b", re.I)
# capability claims about a product/integration: true for *that* customer's product, not necessarily ours
FEATURE_CLAIM = re.compile(r"\b(allows?|supports?|is compatible|integrates? with|offers?|comes with|includes?|"
                           r"ermöglicht|unterstützt|ist kompatibel|bietet|beinhaltet|lässt sich)\b", re.I)
# sentences that only make sense with their original context
ANAPHORA = re.compile(r"^(this|these|that|those|it|they|for example|e\.g\.|additionally|also|furthermore|moreover|"
                      r"dies|diese[rsmn]?|das|es|außerdem|zusätzlich|zum beispiel|ferner)\b|\b(if|whether) (this|that)\b|"
                      r"\bob (dies|das)\b", re.I)
STOP_CAPS = set("I We Our Please Kindly If To For The This In Once After Should Could Would Can May Let Feel You Your "
                "Additionally Also Meanwhile Then Thank Thanks Sie Ihr Ihre Ihnen Wir Bitte Falls Um Sollten Nach "
                "Gerne Damit Zudem Außerdem Support Team IT".split())
ENTITY_TOKEN = re.compile(r"\b(?:[A-Z]{2,}[a-z]*\w*|[A-Za-z]*\d+[A-Za-z\d.]*|[A-Z][a-z]+[A-Z]\w*)\b")  # VPN, Win10, RapidMiner

OPEN = {"en": "Thank you for contacting us. Based on how we resolved similar cases:",
        "de": "Vielen Dank für Ihre Nachricht. Auf Grundlage vergleichbarer, bereits gelöster Fälle:"}
CLOSE = {"en": "If the issue persists, please reply to this message and our team will continue to assist you.",
         "de": "Sollte das Problem weiterhin bestehen, antworten Sie bitte auf diese Nachricht – unser Team unterstützt Sie gerne weiter."}
NO_PRECEDENT = {"en": "Thank you for contacting us. We could not find a comparable resolved case, so a specialist will review your request and get back to you.",
                "de": "Vielen Dank für Ihre Nachricht. Wir haben keinen vergleichbaren gelösten Fall gefunden. Ein Spezialist prüft Ihre Anfrage und meldet sich bei Ihnen."}
FOLLOW_UP = {"en": "Regarding your request for {what}, a specialist will follow up with you; this message does not confirm any amounts, guarantees or commitments.",
             "de": "Zu Ihrer Anfrage bezüglich {what} meldet sich ein Spezialist bei Ihnen; diese Nachricht bestätigt keine Beträge, Garantien oder Zusagen."}
WHAT = {"refund / compensation demand": {"en": "a refund or compensation", "de": "einer Erstattung oder Entschädigung"},
        "legal / contractual guarantee": {"en": "legal or contractual guarantees", "de": "rechtlicher oder vertraglicher Zusagen"},
        "pricing / discount commitment": {"en": "pricing or discounts", "de": "Preisen oder Rabatten"},
        "credential / account takeover": {"en": "passwords or security settings", "de": "Passwörtern oder Sicherheitseinstellungen"}}

MAX_SENTENCES = 4
MIN_SENT_SIM = 0.80      # e5 cosine between ticket and a candidate sentence
REDUNDANT_SIM = 0.92


@lru_cache(maxsize=2)
def _mt(direction: str):
    from transformers import MarianMTModel, MarianTokenizer
    name = f"Helsinki-NLP/opus-mt-{direction}"
    return MarianTokenizer.from_pretrained(name), MarianMTModel.from_pretrained(name).eval()


def translate(sentences, src, tgt):
    if not sentences or src == tgt:
        return list(sentences)
    try:
        import torch
        tok, model = _mt(f"{src}-{tgt}")
        with torch.no_grad():
            out = model.generate(**tok(sentences, return_tensors="pt", padding=True, truncation=True),
                                 max_new_tokens=160, num_beams=2)
        return tok.batch_decode(out, skip_special_tokens=True)
    except Exception:
        return None     # translation unavailable -> caller skips those sentences


def foreign_entities(sentence: str, lang: str, ticket_lower: str) -> list:
    """Product names / acronyms / versions in a precedent sentence that the customer never mentioned.
    Reusing them would 'invent' integrations or products, so such sentences are dropped."""
    toks = set(ENTITY_TOKEN.findall(sentence))
    if lang == "en":   # in English, mid-sentence capitalised words are names (German capitalises all nouns)
        words = sentence.split()[1:]
        toks |= {w.strip(".,;:!?()\"'") for w in words if w[:1].isupper()}
    return [t for t in toks if t and t not in STOP_CAPS and t.lower() not in ticket_lower]


def candidate_sentences(precedents, ticket_text=""):
    cands, ticket_lower = [], ticket_text.lower()
    for rank, p in enumerate(precedents):
        for pos, s in enumerate(SENT_SPLIT.split(p["answer"])):
            s = " ".join(s.split())
            if (len(s.split()) < 5 or not s[0].isupper() or PLACEHOLDER.search(s) or GREETING.search(s)
                    or COMMITMENT.search(s) or ACK.search(s) or CUSTOMER_VOICE.search(s)
                    or FEATURE_CLAIM.search(s) or ANAPHORA.search(s)):
                continue
            s_lang = detect_language("", s, default=p["language"])["lang"]
            if foreign_entities(s, s_lang, ticket_lower):
                continue
            cands.append({"text": s, "ticket_id": p["ticket_id"], "rank": rank, "pos": pos,
                          "prec_sim": p["similarity"], "lang": s_lang})
    return cands


def compose(subject: str, body: str, precedents: list, lang: str, risky: list | None = None,
            min_precedent_sim: float = 0.0) -> dict:
    from retriever import embed
    risky = risky or []
    usable = [p for p in precedents if p["similarity"] >= min_precedent_sim]
    cands = candidate_sentences(usable, f"{subject}\n{body}")

    chosen = []
    if cands:
        q = embed([f"{subject}\n{body}".strip()])[0]
        E = embed([c["text"] for c in cands])
        score = (E @ q) * (0.5 + 0.5 * np.array([c["prec_sim"] for c in cands]))
        for i in np.argsort(-score):
            if float(E[i] @ q) < MIN_SENT_SIM:
                break
            if any(float(E[i] @ E[j]) >= REDUNDANT_SIM for j in chosen):
                continue
            chosen.append(int(i))
            if len(chosen) == MAX_SENTENCES:
                break
        chosen.sort(key=lambda i: (cands[i]["pos"] > 0, cands[i]["rank"], cands[i]["pos"]))

    picked = [dict(cands[i]) for i in chosen]
    for src in ("en", "de"):
        idx = [k for k, c in enumerate(picked) if c["lang"] == src and src != lang]
        if idx:
            tr = translate([picked[k]["text"] for k in idx], src, lang)
            for k, t in zip(idx, tr or [None] * len(idx)):
                picked[k]["translated_from"] = src
                picked[k]["text"] = t
    picked = [c for c in picked if c["text"]]

    if not picked:
        lines = [NO_PRECEDENT[lang]]
    else:
        lines = [OPEN[lang], ""] + [f"• {c['text'].rstrip()} [{c['ticket_id']}]" for c in picked]
    for r in risky:
        if r in WHAT:
            lines += ["", FOLLOW_UP[lang].format(what=WHAT[r][lang])]
    if picked:
        lines += ["", CLOSE[lang]]

    cited = list(dict.fromkeys(c["ticket_id"] for c in picked))
    return {"reply": "\n".join(lines), "cited_ticket_ids": cited,
            "evidence_sufficient": bool(picked),
            "unsupported_requests": [],
            "sentences": [{"ticket_id": c["ticket_id"], "text": c["text"],
                           "translated_from": c.get("translated_from")} for c in picked]}
