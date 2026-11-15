"""Discover only the named crop datasets; never scan the project venv or models."""
from __future__ import annotations
import hashlib,os,re
from pathlib import Path
from collections import Counter
from PIL import Image,UnidentifiedImageError
IMAGE_EXTS={'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}
DATASET_NAMES=('bananadisease','corndisease','mangodisease','millet disease','non-leaf data','Pomegranate Diseases','rice_leaf_diseases','soyabeandisease','sugarcane disease','tomatodisease')
DEFAULT_ROOT=Path(os.environ.get('DISEASE_DATASET',r'C:\Users\asus\Desktop\diseasedetect'))
ALIASES={'banana':'Banana','corn':'Corn','maize':'Corn','mango':'Mango','millet':'Millet','pomegranate':'Pomegranate','rice':'Rice','soyabean':'Soybean','soybean':'Soybean','sugarcane':'Sugarcane','tomato':'Tomato'}
HEALTHY=re.compile(r'healthy|healty|health|normal|clean|without disease',re.I)
def crop_name(path:Path,root:Path)->str:
 for part in path.relative_to(root).parts[:-1]:
  for alias,name in ALIASES.items():
   if alias in part.lower():return name
 return 'Unknown'
def disease_name(path:Path)->str:
 name=path.parent.name.replace('_',' ').replace('-',' ').strip()
 name=re.sub(r'^(banana|corn \(maize\)|corn|mango|millet|pomegranate|rice|soyabean|soybean|sugarcane|suagrcane|tomato)\s*','',name,flags=re.I)
 name=re.sub(r'\s+',' ',name).strip(' _-')
 if HEALTHY.search(name):return 'Healthy'
 return name.title() if name else path.parent.name
def iter_samples(root:Path=DEFAULT_ROOT):
 root=Path(root)
 if not root.is_dir():raise FileNotFoundError(f'Dataset folder not found: {root}')
 children={p.name.lower():p for p in root.iterdir() if p.is_dir()}; dataset_roots=[]; missing=[]
 for name in DATASET_NAMES:
  path=children.get(name.lower())
  if path is None:missing.append(name)
  else:dataset_roots.append(path)
 bad=[];seen={};duplicates=0;conflicts=0;counts=Counter()
 for base in dataset_roots:
  negative=base.name.lower().replace(' ','-')=='non-leaf-data'
  for path in sorted(base.rglob('*')):
   if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:continue
   try:
    with Image.open(path) as im:im.verify()
   except (OSError,UnidentifiedImageError,ValueError):bad.append(str(path));continue
   digest=hashlib.sha256()
   with path.open('rb') as f:
    for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
   key=digest.hexdigest()
   if negative:crop='';disease='';label='Non-Leaf'
   else:crop=crop_name(path,root);disease=disease_name(path);label=f'{crop} / {disease}'
   if key in seen:
    duplicates+=1
    if seen[key]!=label:conflicts+=1
    continue
   seen[key]=label
   if negative:yield {'path':path,'leaf':False,'crop':'','disease':''};counts['Non-Leaf']+=1
   else:yield {'path':path,'leaf':True,'crop':crop,'disease':disease};counts[label]+=1
 print(f'Valid unique images: {len(seen)}')
 for name,count in sorted(counts.items()):print(f'  {name}: {count}')
 if duplicates:print(f'Exact duplicate files skipped: {duplicates}')
 if conflicts:print(f'Exact duplicates found under conflicting class labels; first sorted path retained: {conflicts}')
 if bad:print(f'Unreadable supported image files skipped: {len(bad)}')
 if missing:print('Expected dataset folders missing: '+', '.join(missing))

