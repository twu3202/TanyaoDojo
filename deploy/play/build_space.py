"""Allowlisted HF build context. Does not publish or read credentials."""
import argparse,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--assets',required=True);p.add_argument('--native-source',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2];out=Path(a.output).resolve();assets=Path(a.assets);native=Path(a.native_source)
if out.exists() and any(out.iterdir()):raise SystemExit('output must be a new empty directory')
out.mkdir(parents=True,exist_ok=True)
for rel in ['deploy/play','sichuan','common']:
 source=root/rel;dest=out/rel;dest.mkdir(parents=True,exist_ok=True)
 for f in source.rglob('*'):
  if not f.is_file() or '__pycache__' in f.parts:continue
  if rel!='deploy/play' and f.name not in ['reference_impl.py','suit_table.py','RULES.md']:continue
  if rel=='deploy/play' and ('tests' in f.parts or f.name in ['build_space.py']):continue
  if f.suffix not in ('.py','.md','.html','.js','.css','.svg','.txt') and f.name!='Dockerfile':continue
  target=dest/f.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
for src,dst in [('SPACE_README.md','README.md'),('Dockerfile','Dockerfile'),('requirements.txt','requirements.txt')]:shutil.copy2(root/'deploy/play'/src,out/dst)
(out/'assets').mkdir()
for name in ['model.pth','model.py','sichuan.npz']:shutil.copy2(assets/name,out/'assets'/name)
# Ship and compile the exact native source, rather than copying a platform-specific opaque .so.
ns=out/'native_source';ns.mkdir()
for name in ['Cargo.toml','Cargo.lock','LICENSE']:shutil.copy2(native/name,ns/name)
for package in ['libriichi','exe-wrapper']:
 dest=ns/package;dest.mkdir()
 shutil.copy2(native/package/'Cargo.toml',dest/'Cargo.toml')
 if (native/package/'build.rs').exists():shutil.copy2(native/package/'build.rs',dest/'build.rs')
 shutil.copytree(native/package/'src',dest/'src')
 if (native/package/'benches').exists():shutil.copytree(native/package/'benches',dest/'benches')
shutil.copy2(native/'LICENSE',out/'LICENSE');shutil.copy2(root/'LICENSE',out/'LICENSE-MIT')
manifest={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in out.rglob('*') if f.is_file()}
(out/'ASSET_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
(out/'.gitattributes').write_text('*.pth filter=lfs diff=lfs merge=lfs -text\n*.npz filter=lfs diff=lfs merge=lfs -text\n')
print('build context:',out,'files:',len(manifest),'bytes:',sum(f.stat().st_size for f in out.rglob('*') if f.is_file()))
