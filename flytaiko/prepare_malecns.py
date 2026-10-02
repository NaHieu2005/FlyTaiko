"""Fetch official, public MaleCNS v1.0 tables; never fall back to synthetic data."""
import hashlib
import json
from pathlib import Path
import subprocess
import pyarrow.feather as feather

ROOT = Path('connectome_data/malecns_v1')
BASE = 'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'
FILES = ['body-annotations-male-cns-v1.0-minconf-0.5.feather',
         'body-neurotransmitters-male-cns-v1.0.feather',
         'connectome-weights-male-cns-v1.0-minconf-0.5.feather']

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    records = {}
    for name in FILES:
        path = ROOT / name
        if not path.exists():
            partial = path.with_suffix('.download')
            subprocess.run(['curl','--fail','--location','--retry','5','--continue-at','-',
                            '--output',str(partial),BASE+name],check=True)
            feather.read_table(partial, memory_map=True)
            partial.replace(path)
        table=feather.read_table(path,memory_map=True)
        with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
        records[name]={'url':BASE+name,'bytes':path.stat().st_size,'sha256':digest,
                       'rows':table.num_rows,'schema':str(table.schema)}
        print(json.dumps({name:records[name]}),flush=True)
    manifest={'dataset':'MaleCNS v1.0','license':'CC-BY-4.0',
              'source':'https://male-cns.janelia.org/download/',
              'hash_note':'Locally computed download hashes, not publisher signatures', 'files':records}
    target=ROOT/'source_manifest.json';temporary=target.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest,indent=2));temporary.replace(target)

if __name__=='__main__':main()
