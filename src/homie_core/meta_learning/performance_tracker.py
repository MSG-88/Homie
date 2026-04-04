# src/homie_core/meta_learning/performance_tracker.py
"""Meta Performance Tracker -- strategy-aware, with SQLite persistence."""
from __future__ import annotations
import logging, time
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)
_DAY = 86400

@dataclass
class _E:
    task_type: str; timestamp: float; duration_ms: float; success: bool
    quality_score: float; strategy_key: str = ""; satisfaction: float = 0.0

class MetaPerformanceTracker:
    def __init__(self, storage=None):
        self._storage, self._entries = storage, []
        if storage:
            try:
                for r in storage.load_task_entries(limit=1000):
                    self._entries.append(_E(task_type=r["task_type"], timestamp=0.0,
                        duration_ms=r["duration_ms"], success=r["success"],
                        quality_score=r["quality_score"], strategy_key=r.get("strategy_key",""),
                        satisfaction=r.get("satisfaction",0.0)))
            except: pass

    def record_task(self, task_type, duration_ms, success, quality_score,
                    strategy_key="", satisfaction=0.0, context=None):
        e = _E(task_type=task_type, timestamp=time.time(), duration_ms=duration_ms,
               success=success, quality_score=max(0.0,min(1.0,quality_score)),
               strategy_key=strategy_key, satisfaction=max(0.0,min(1.0,satisfaction)))
        self._entries.append(e)
        if self._storage:
            try: self._storage.add_task_entry(task_type=task_type, strategy_key=strategy_key, duration_ms=duration_ms, success=success, quality_score=e.quality_score, satisfaction=e.satisfaction, context=context)
            except: pass

    def get_improvement_trend(self, task_type, window_days=30):
        cutoff = time.time() - window_days*_DAY
        rel = sorted([e for e in self._entries if e.task_type==task_type and e.timestamp>=cutoff], key=lambda e: e.timestamp)
        if len(rel)<2: return {"task_type":task_type,"direction":"insufficient_data","improvement_rate":0.0,"confidence":0.0,"sample_size":len(rel)}
        m=len(rel)//2; d=0.6*(_sr(rel[m:])-_sr(rel[:m]))+0.4*(_aq(rel[m:])-_aq(rel[:m]))
        return {"task_type":task_type,"direction":"improving" if d>0.02 else ("declining" if d<-0.02 else "stable"),
                "improvement_rate":round(d,4),"confidence":round(min(1.0,len(rel)/50),4),"sample_size":len(rel)}

    def get_strategy_performance(self, task_type):
        by = {}
        for e in self._entries:
            if e.task_type==task_type and e.strategy_key: by.setdefault(e.strategy_key,[]).append(e)
        return {k:{"attempts":len(v),"success_rate":round(_sr(v),4),"avg_quality":round(_aq(v),4),
                    "avg_satisfaction":round(_asat(v),4),"avg_duration_ms":round(sum(e.duration_ms for e in v)/len(v),2)} for k,v in by.items()}

    def get_best_strategy(self, task_type):
        p=self.get_strategy_performance(task_type)
        return max(p,key=lambda k:0.4*p[k]["success_rate"]+0.35*p[k]["avg_quality"]+0.25*p[k]["avg_satisfaction"]) if p else None

    def get_bottlenecks(self):
        by={}
        for e in self._entries: by.setdefault(e.task_type,[]).append(e)
        bn=[]
        for tt,ents in by.items():
            sr,aq=_sr(ents),_aq(ents); sc=0.6*sr+0.4*aq
            if sc<0.7: bn.append({"task_type":tt,"success_rate":round(sr,4),"avg_quality":round(aq,4),"composite_score":round(sc,4),"sample_size":len(ents),"best_strategy":self.get_best_strategy(tt)})
        bn.sort(key=lambda b:b["composite_score"]); return bn

    def get_overall_health(self):
        if not self._entries: return {"status":"no_data","total_tasks":0,"overall_success_rate":0.0,"overall_avg_quality":0.0,"bottleneck_count":0}
        sr,aq=_sr(self._entries),_aq(self._entries)
        st="healthy" if sr>=0.9 and aq>=0.8 else ("fair" if sr>=0.7 else "needs_attention")
        return {"status":st,"total_tasks":len(self._entries),"overall_success_rate":round(sr,4),"overall_avg_quality":round(aq,4),"bottleneck_count":len(self.get_bottlenecks())}

def _sr(e): return sum(1 for x in e if x.success)/len(e) if e else 0.0
def _aq(e): return sum(x.quality_score for x in e)/len(e) if e else 0.0
def _asat(e):
    v=[x.satisfaction for x in e if x.satisfaction>0]; return sum(v)/len(v) if v else 0.0
