"""Interpret the archived diagnostic columns using the manuscript's notation."""
import argparse,csv,json,math,statistics
from pathlib import Path

def crossfit_statistics(loss,g2,cross):
    if loss<=0:raise ValueError('The full-KL rate factorization requires positive loss')
    return {'mu':g2/(2*loss),'alpha':cross/g2 if g2!=0 else None,
            'gamma':(g2+cross)/(2*loss),'positive_g2':g2>0}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('csv',type=Path);a=p.parse_args()
    rows=[]
    for r in csv.DictReader(a.csv.open()):
        rows.append({'teacher':r['teacher'],'step':int(r['step']),**crossfit_statistics(float(r['F_top16']),float(r['g2_crossfit']),float(r['X_dot']))})
    print(json.dumps(rows,indent=2,allow_nan=False))
if __name__=='__main__':main()
