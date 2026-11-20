import argparse,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def domains(n=1200,seed=27):
 r=np.random.default_rng(seed);w=r.normal(size=12);xs=r.normal(size=(n,12));ys=r.binomial(1,1/(1+np.exp(-(xs@w/3))));A=np.eye(12)+r.normal(scale=.12,size=(12,12));xt=r.normal(size=(n,12))@A+.35;yt=r.binomial(1,1/(1+np.exp(-((xt-.35)@w/3))));return xs,ys,xt,yt
def root(c,power):
 v,u=np.linalg.eigh(c+np.eye(c.shape[0])*1e-5);return (u*(v**power))@u.T
def evaluate(seed=27):
 xs,ys,xt,yt=domains(seed=seed);base=LogisticRegression(max_iter=500).fit(xs,ys);aligned=(xt-xt.mean(0))@root(np.cov(xt,rowvar=False),-.5)@root(np.cov(xs,rowvar=False),.5)+xs.mean(0);return {"source_only_target_auroc":float(roc_auc_score(yt,base.predict_proba(xt)[:,1])),"coral_target_auroc":float(roc_auc_score(yt,base.predict_proba(aligned)[:,1])),"source_auroc":float(roc_auc_score(ys,base.predict_proba(xs)[:,1]))}
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("outputs/demo"));p.add_argument("--seed",type=int,default=27);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);r=evaluate(a.seed);(a.output/"metrics.json").write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=="__main__":main()
