"""Reparse the SAME selected native songs, verifying original file hashes."""
import argparse
import hashlib
import json
from pathlib import Path
from taiko.parser import parse_osu_text
from flytaiko.extract_beatmaps import extract_beatmap_data
from flytaiko.neural_campaign import atomic_json

def refresh(row):
    source=Path(row['source']);raw=source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=row['source_sha256']:raise ValueError('Source changed: '+str(source))
    bm=parse_osu_text(raw.decode('utf-8-sig'))
    if bm is None:raise ValueError('Not native taiko: '+str(source))
    return {**row,**extract_beatmap_data(bm)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    old=json.loads(a.source.read_text());new={k:[refresh(r) for r in rows] for k,rows in old.items()}
    changes={k:[{'index':i,'title':r['metadata']['title'],
        'changed_notes':sum(x!=y for x,y in zip(r['notes'],old[k][i]['notes']))}
        for i,r in enumerate(rows) if r!=old[k][i]] for k,rows in new.items()}
    atomic_json(a.output,new);atomic_json(a.output.with_name(a.output.stem+'-changes.json'),changes)
    print(json.dumps({'event':'native_split_refreshed','changes':changes}),flush=True)

if __name__=='__main__':main()
