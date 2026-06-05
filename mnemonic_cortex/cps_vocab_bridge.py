from typing import Dict, List

from .cps import ConsolidatedParamStore, UnifiedParam, UnifiedParamCfg


def key_token(s: str) -> str:
    return f"token:{s}"


def key_skill(s: str) -> str:
    return f"skill:{s}"


def key_entity(s: str) -> str:
    return f"entity:{s}"


def ensure_tokens(cps: ConsolidatedParamStore, toks: List[str], cfg: UnifiedParamCfg = None) -> Dict[str, UnifiedParam]:
    out: Dict[str, UnifiedParam] = {}
    for t in toks:
        out[t] = cps.ensure(key_token(t), cfg)
    return out


def ensure_skills(cps: ConsolidatedParamStore, skills: List[str], cfg: UnifiedParamCfg = None) -> Dict[str, UnifiedParam]:
    out: Dict[str, UnifiedParam] = {}
    for s in skills:
        out[s] = cps.ensure(key_skill(s), cfg)
    return out


def ensure_entities(cps: ConsolidatedParamStore, ents: List[str], cfg: UnifiedParamCfg = None) -> Dict[str, UnifiedParam]:
    out: Dict[str, UnifiedParam] = {}
    for e in ents:
        out[e] = cps.ensure(key_entity(e), cfg)
    return out

