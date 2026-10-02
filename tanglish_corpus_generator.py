#!/usr/bin/env python3
"""
Phase 1B: Synthetic Tanglish, Dialects & English Corpus Generator
Generates high-density Romanized Telugu across Coastal Andhra, Telangana, 
and Rayalaseema dialects, along with common English/Tech vocabulary.
"""

import os
import random

OUTPUT_FILE = "tanglish_dialect_corpus.txt"
TARGET_SENTENCES = 1_000_000

# Dialect & Tech Vocabulary
VERBS = {
    "doing": ["chestunnanu", "chestunna", "yestunna", "yesta", "sesthannam", "doing", "chestunnav"],
    "coming": ["vostunnanu", "vostunna", "ostha", "yastna", "osthunna", "vasthuna", "vosthana"],
    "saw": ["chusa", "chusanu", "soosa", "soosina", "chushina", "chusina", "chuse"],
    "went": ["poyina", "poyanu", "poyinav", "velanu", "poyinru", "vellostanu"],
    "eat": ["thinte", "thintunna", "thinesa", "thinnava", "thinnara"],
    "knowing": ["telusu", "telusthundi", "eruka", "yeruka", "thelusura"]
}

DIALECT_MARKERS = {
    "telangana": ["yada", "yendi", "kirrak", "zabarjast", "masth", "baigan", "gatlana", "itlana", "haula", "malla", "eppuno", "emmi"],
    "andhra": ["yeti", "abbaba", "andi", "emandi", "garu", "baga", "chala", "mari", "itiga", "bagunnara"],
    "rayalaseema": ["saami", "yami", "ra", "bava", "nayana", "agudam", "poga", "yara", "kaludam", "pokunda"]
}

TEMPLATES = [
    "emmi doing ra baigan? yada poyinav eppuno chusa.",
    "eti chestunnav andi? bagunnara?",
    "yami saami agudam pokunda kaludam.",
    "API response load avvaledhu, server restart chesa.",
    "nenu {verb_doing} work, nuvvu {verb_doing} work ah?",
    "aah {verb_saw} video, chala {adj} undhi bro.",
    "project status {status} chesa, repo push {verb_went}.",
    "malli {verb_coming} mama, wait cheyi 5 minutes.",
    "naku exact ga {verb_knowing}, clear ga explain cheyyi.",
    "{dialect_tg} poyinav ra? eppudu {verb_coming} intiki?",
    "{dialect_ap} chestunnav erra? call cheste lift cheyavu.",
    "{dialect_seema} entha sepu wait cheyali? jaldi ra.",
    "server restart {verb_doing}, issue resolve authundhi.",
    "bug fix {verb_saw}, PR review chesi merge cheyyi.",
    "today meeting lo discuss {verb_doing} topics gurinchi email pampu.",
    "naa machine lo code run {verb_doing}, zero errors vachayi.",
    "emmi doing bro? simple ga chilling output vundhi.",
    "dinner {verb_eat} ah? em special eeroju?",
    "chusa kani reply ivvaledhu kavalani.",
    "office ki {verb_went} tharvatha ping chestha."
]

ADJECTIVES = ["good", "super", "kirrak", "heavy", "bad", "slow", "fast", "awesome"]
STATUSES = ["update", "submit", "deploy", "complete", "cancel"]

def generate_tanglish():
    print("=" * 70)
    print("🌐 GENERATING MULTIDIALECT TANGLISH & CODE-SWITCHED CORPUS (1M Sentences)")
    print("=" * 70)
    
    sentences = []
    
    for _ in range(TARGET_SENTENCES):
        tmpl = random.choice(TEMPLATES)
        
        v_doing = random.choice(VERBS["doing"])
        v_saw = random.choice(VERBS["saw"])
        v_went = random.choice(VERBS["went"])
        v_coming = random.choice(VERBS["coming"])
        v_eat = random.choice(VERBS["eat"])
        v_knowing = random.choice(VERBS["knowing"])
        
        sentence = tmpl.format(
            verb_doing=v_doing,
            verb_saw=v_saw,
            verb_went=v_went,
            verb_coming=v_coming,
            verb_eat=v_eat,
            verb_knowing=v_knowing,
            adj=random.choice(ADJECTIVES),
            status=random.choice(STATUSES),
            dialect_tg=random.choice(DIALECT_MARKERS["telangana"]),
            dialect_ap=random.choice(DIALECT_MARKERS["andhra"]),
            dialect_seema=random.choice(DIALECT_MARKERS["rayalaseema"])
        )
        sentences.append(sentence)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(sentences))
        
    print(f"✓ Generated {len(sentences):,} sentences into '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    generate_tanglish()