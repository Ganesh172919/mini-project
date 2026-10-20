"""Create a portable archive with only the two-approach submission and required inputs."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'submission'
def main():
    dest=OUT/'deliverables/Submission_Package.zip'
    files=[]
    for p in OUT.rglob('*'):
        if not p.is_file():continue
        parts=p.relative_to(OUT).parts
        if any(v in ['.build','.git','__pycache__','node_modules'] for v in parts):continue
        if p==dest or p.name=='package_manifest.json':continue
        if p.suffix in ['.pyc','.tmp','.log']:continue
        files.append(p)
    files.append(ROOT/'Health Insurance Fraud Claims.xlsx')
    for folder in ['Approch-1/models','Approch-1/data/processed','Approch-1/data/raw','Approch-2-DL/models/mlp']:
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file())
    files.extend(ROOT/'Approch-2-DL/scripts'/n for n in ['dl_common.py','submission_models.py'])
    files=sorted(set(files))
    records=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
    manifest=dict(file_count=len(records),files=records)
    (OUT/'results/package_manifest.json').write_text(json.dumps(manifest,indent=2))
    with ZipFile(dest,'w',ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,p.relative_to(ROOT).as_posix())
        z.write(OUT/'results/package_manifest.json','submission/results/package_manifest.json')
    with ZipFile(dest) as z:
        assert z.testzip() is None
        for record in records:assert hashlib.sha256(z.read(record['path'])).hexdigest()==record['sha256']
        assert not any('Approach-3' in n or '.venv/' in n or '/.git/' in n for n in z.namelist())
    print(json.dumps(dict(archive=str(dest),files=len(records)+1,bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest()),indent=2))

if __name__=='__main__':main()
