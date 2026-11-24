"""Small status receipts and notebook progress; detailed output stays in log files."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from smolvlm_attention.sample_store import atomic_json

class StatusWriter:
    def __init__(self,output,rank=None):
        self.root=Path(output)/'status';self.root.mkdir(parents=True,exist_ok=True)
        self.rank=int(os.environ.get('RANK','0')) if rank is None else rank
        self.last=0.;self.phase=None
    def update(self,phase,completed,total,*,force=False,**details):
        now=time.monotonic()
        if not force and phase==self.phase and now-self.last<2.:return
        atomic_json(self.root/f'rank_{self.rank}.json',dict(phase=phase,rank=self.rank,completed=int(completed),total=int(total),updated_at=time.time(),**details))
        self.last=now;self.phase=phase

class ProjectionProgress:
    """Durably committed groups; resumed groups never inflate compute throughput."""
    def __init__(self,output,rank,total=None):
        self.status=StatusWriter(output,rank);self.started=time.perf_counter()
        self.previous=0;self.resumed=0
        if total is not None:self.status.update('projected_cache',0,total,force=True)
    def __call__(self,value):
        done,total=value['committed'],value['total']
        if value.get('reused',False):self.resumed+=done-self.previous
        self.previous=done
        elapsed=max(time.perf_counter()-self.started,1e-9)
        newly=done-self.resumed;rate=newly/elapsed if newly else None
        self.status.update('projected_cache',done,total,force=done==total,
            elapsed_seconds=elapsed,committed_new=newly,resumed_groups=self.resumed,
            groups_per_second=rate,estimated_remaining_seconds=(total-done)/rate if rate else None)


def cache_postfix(peers):
    elapsed=max((x.get('elapsed_seconds',0.) for x in peers),default=0.)
    newly=sum(x.get('committed_new',0) for x in peers)
    rate=newly/elapsed if elapsed>0 and newly else None
    estimates=[x.get('estimated_remaining_seconds') for x in peers if x['completed']<x['total']]
    eta=max(estimates) if estimates and all(x is not None for x in estimates) else (0. if not estimates else None)
    def duration(seconds):
        if seconds is None:return '--'
        seconds=int(seconds);return f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}'
    return f"elapsed {duration(elapsed)}, {rate:.2f} groups/s, ETA {duration(eta)}" if rate else f"elapsed {duration(elapsed)}, ETA --"


def latest_phase(statuses):
    """Do not replay a lagging rank's earlier phase after a phase transition."""
    if not statuses:
        return []
    priority={'projected_cache_validation':-4,'projected_cache_preflight':-3,
              'projected_cache':-2,'projected_cache_finalize':-1,'train':0,'validation':1,'test':2}
    key=lambda x:(x.get('epoch') or 0,priority.get(x['phase'],-1))
    current=max(statuses,key=key)
    return [x for x in statuses if key(x)==key(current) and (x['phase']!='train' or x['rank']==0)]


def _bar(total,desc,unit):
    from IPython import get_ipython
    if get_ipython() is not None:
        try:
            from tqdm.notebook import tqdm
            return tqdm(total=total,desc=desc,unit=unit,leave=True)
        except (ImportError,RuntimeError):
            from IPython.display import display
            class DisplayProgress:
                def __init__(self):
                    self.total=total; self.n=0; self.postfix=''; self.handle=display('',display_id=True)
                def refresh(self): self.handle.update(f'{desc}: {self.n}/{self.total} {unit} {self.postfix}')
                def set_postfix_str(self,text,refresh=False):
                    self.postfix=text
                    if refresh:self.refresh()
                def close(self): self.refresh()
            return DisplayProgress()
    from tqdm import tqdm
    return tqdm(total=total,desc=desc,unit=unit,leave=True)


def run_notebook(module,command,args,output,source):
    """One widget per phase; detailed child output stays entirely in log files."""
    output=Path(output);logs=output/'logs';logs.mkdir(parents=True,exist_ok=True)
    logfile=logs/f'notebook_{command}.log';started=time.time();bars={};process=None
    try:
        with logfile.open('a',encoding='utf-8') as stream:
            process=subprocess.Popen([sys.executable,'-m',module,command,*map(str,args)],cwd=source,stdout=stream,stderr=subprocess.STDOUT)
            while True:
                statuses=[]
                for path in sorted((output/'status').glob('rank_*.json')):
                    try:value=json.loads(path.read_text(encoding='utf-8'))
                    except (OSError,ValueError):continue
                    if value.get('updated_at',0)>=started:statuses.append(value)
                peers=latest_phase(statuses)
                if peers:
                    value=peers[0];key=(value['phase'],value.get('epoch'))
                    total=sum(x['total'] for x in peers);done=sum(x['completed'] for x in peers)
                    if key not in bars:
                        for old in bars.values():old.close()
                        desc=value['phase']+(f" epoch {value['epoch']}/5" if value.get('epoch') else '')
                        unit='batch' if value['phase']=='train' else ('group' if value['phase']=='projected_cache' else ('step' if value['phase'].startswith('projected_cache_') else 'sample'))
                        bars[key]=_bar(total,desc,unit)
                        if value['phase']=='projected_cache':
                            bars[key].bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} groups [{postfix}]'
                    bar=bars[key]
                    if value['phase']=='projected_cache':
                        bar.set_postfix_str(cache_postfix(peers),refresh=False)
                    if bar.total!=total or bar.n!=done or value['phase']=='projected_cache':
                        bar.total=total;bar.n=done;bar.refresh()
                status=process.poll()
                if status is not None:
                    if status:raise RuntimeError(f'{command} failed (exit {status}); see {logfile}')
                    break
                time.sleep(1.)
    finally:
        for bar in bars.values():bar.close()
        if process is not None and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:process.kill();process.wait()
    return logfile
