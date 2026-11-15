"""Print a dataset inventory and validation summary without copying files."""
import argparse
from pathlib import Path
from collections import Counter
from dataset import DEFAULT_ROOT, iter_samples

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,default=DEFAULT_ROOT); a=p.parse_args()
    items=list(iter_samples(a.data)); print(f'\nDataset root: {a.data.resolve()}'); print(f'Valid unique images: {len(items)}'); print('Detected leaf classes:')
    counts=Counter((x['crop'],x['disease']) for x in items if x['leaf'])
    for (crop,disease),n in sorted(counts.items()): print(f'  {crop} / {disease}: {n}')
    print(f"Non-leaf examples: {sum(not x['leaf'] for x in items)}")
if __name__=='__main__': main()
