# src/homie_core/meta_learning/auto_tuner.py
"""Bayesian-inspired hyperparameter tuning with SQLite persistence."""
from __future__ import annotations
import logging, random, time
from dataclasses import dataclass, field
from typing import Any
from .performance_tracker import MetaPerformanceTracker

log = logging.getLogger(__name__)

@dataclass
class BetaPrior:
    alpha: float = 1.0; beta: float = 1.0
    @property
    def mean(self): return self.alpha/(self.alpha+self.beta)
    @property
    def variance(self): a,b=self.alpha,self.beta; return (a*b)/((a+b)**2*(a+b+1))
    @property
    def samples(self): return int(self.alpha+self.beta-2)
    def update(self, success):
        if success: self.alpha += 1.0
        else: self.beta += 1.0
    def sample(self): return random.betavariate(max(self.alpha,0.01),max(self.beta,0.01))
    def to_dict(self): return {"alpha":self.alpha,"beta":self.beta}
    @classmethod
    def from_dict(cls, d): return cls(alpha=d.get("alpha",1.0),beta=d.get("beta",1.0))

@dataclass
class TunableParam:
    name: str; value: float; min_val: float; max_val: float; step: float
    prior: BetaPrior = field(default_factory=BetaPrior)
    def propose_change(self):
        belief=self.prior.sample()
        delta=random.uniform(-self.step,self.step) if belief>0.6 else random.uniform(-3*self.step,3*self.step)
        return max(self.min_val,min(self.max_val,round(self.value+delta,4)))

_DEFAULTS=[
    {"name":"temperature","value":0.7,"min_val":0.1,"max_val":1.2,"step":0.05},
    {"name":"max_tokens","value":768,"min_val":128,"max_val":4096,"step":64},
    {"name":"context_budget","value":1.0,"min_val":0.3,"max_val":2.0,"step":0.1},
    {"name":"explore_rate","value":0.15,"min_val":0.02,"max_val":0.40,"step":0.02},
]

@dataclass
class _Rec:
    parameter: str; old_value: Any; new_value: Any; reason: str
    applied_at: float; reverted: bool = False; db_id: int|None = None

def _r_cache(cfg,trk):
    hr=cfg.get("cache_hit_rate",0.0); mx=cfg.get("cache_max_entries",500)
    return [{"parameter":"cache_max_entries","old_value":mx,"new_value":min(mx*2,5000),"reason":f"Cache hit {hr:.0%}"}] if hr>=0.8 and mx<5000 else []
def _r_probe(cfg,trk):
    h=trk.get_overall_health(); iv=cfg.get("probe_interval_s",30)
    return [{"parameter":"probe_interval_s","old_value":iv,"new_value":min(iv*2,120),"reason":"Healthy"}] if h.get("status")=="healthy" and iv<120 else []
def _r_explore(cfg,trk):
    sr=trk.get_overall_health().get("overall_success_rate",0.0); e=cfg.get("explore_rate",0.15)
    if sr>=0.9 and e>0.05: return [{"parameter":"explore_rate","old_value":e,"new_value":max(e*0.5,0.05),"reason":f"SR {sr:.0%}"}]
    if sr<0.6 and e<0.30: return [{"parameter":"explore_rate","old_value":e,"new_value":min(e*2,0.30),"reason":f"SR {sr:.0%}"}]
    return []

class AutoTuner:
    """Bayesian-inspired hyperparameter tuner with SQLite persistence."""
    def __init__(self, config, performance_tracker, storage=None):
        self._config,self._tracker,self._storage=config,performance_tracker,storage
        self._history=[]; self._params={}
        for s in _DEFAULTS:
            self._params[s["name"]]=TunableParam(name=s["name"],value=float(config.get(s["name"],s["value"])),
                min_val=s["min_val"],max_val=s["max_val"],step=s["step"])
        if storage:
            try:
                for n,d in storage.load_tuner_params().items():
                    if n in self._params:
                        self._params[n].value=float(d["value"])
                        if d.get("prior"): self._params[n].prior=BetaPrior.from_dict(d["prior"])
                        config[n]=d["value"]
                for r in storage.load_tuner_history():
                    self._history.append(_Rec(parameter=r["parameter"],old_value=r["old_value"],
                        new_value=r["new_value"],reason=r["reason"],applied_at=0.0,
                        reverted=bool(r.get("reverted",False)),db_id=r.get("id")))
            except: pass

    def suggest_tunings(self):
        out=[]
        for rule in [_r_cache,_r_probe,_r_explore]:
            try:
                r=rule(self._config,self._tracker)
                if r: out.extend(r)
            except: pass
        for n,p in self._params.items():
            if p.prior.samples<5: continue
            proposed=p.propose_change()
            if abs(proposed-p.value)>=p.step*0.5:
                out.append({"parameter":n,"old_value":p.value,"new_value":proposed,
                            "reason":f"Bayesian: mean={p.prior.mean:.3f}, n={p.prior.samples}","source":"bayesian"})
        return out

    def apply_tuning(self, tuning):
        pn=tuning.get("parameter"); nv=tuning.get("new_value")
        if not pn: return False
        ov=self._config.get(pn); self._config[pn]=nv
        if pn in self._params: self._params[pn].value=float(nv) if isinstance(nv,(int,float)) else nv
        db_id=None
        if self._storage:
            try: db_id=self._storage.add_tuner_history(parameter=pn,old_value=ov,new_value=nv,reason=tuning.get("reason",""))
            except: pass
        self._history.append(_Rec(parameter=pn,old_value=ov,new_value=nv,reason=tuning.get("reason",""),applied_at=time.time(),db_id=db_id))
        return True

    def record_param_outcome(self, param_name, success):
        if param_name not in self._params: return
        self._params[param_name].prior.update(success)
        if self._storage:
            p=self._params[param_name]
            try: self._storage.upsert_tuner_param(param_name=param_name,value=p.value,prior=p.prior.to_dict())
            except: pass

    def revert_tuning(self, parameter):
        for rec in reversed(self._history):
            if rec.parameter==parameter and not rec.reverted:
                self._config[parameter]=rec.old_value; rec.reverted=True
                if parameter in self._params and isinstance(rec.old_value,(int,float)):
                    self._params[parameter].value=float(rec.old_value)
                if self._storage and rec.db_id:
                    try: self._storage.mark_reverted(rec.db_id)
                    except: pass
                return True
        return False

    def get_tuning_history(self):
        return [{"parameter":r.parameter,"old_value":r.old_value,"new_value":r.new_value,
                 "reason":r.reason,"applied_at":r.applied_at,"reverted":r.reverted} for r in self._history]

    def get_param_beliefs(self):
        return {n:{"value":p.value,"prior_mean":round(p.prior.mean,4),"prior_variance":round(p.prior.variance,6),
                    "samples":p.prior.samples,"range":[p.min_val,p.max_val]} for n,p in self._params.items()}
