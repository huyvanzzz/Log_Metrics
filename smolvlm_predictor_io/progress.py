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

def latest_phase(statuses):
    """Do not replay a lagging rank's earlier phase after a phase transition."""
    if not statuses:
        return []
    priority={'train':0,'validation':1,'test':2}
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
                    self.total=total; self.n=0; self.handle=display('',display_id=True)
                def refresh(self): self.handle.update(f'{desc}: {self.n}/{self.total} {unit}')
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
                        bars[key]=_bar(total,desc,'batch' if value['phase']=='train' else 'sample')
                    bar=bars[key]
                    if bar.total!=total or bar.n!=done:
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
