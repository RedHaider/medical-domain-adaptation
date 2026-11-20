from domain_adapt.experiment import domains,evaluate
def test_shape():
 a,b,c,d=domains(20,2);assert a.shape==c.shape==(20,12) and len(b)==len(d)==20
def test_metrics():assert all(0<=v<=1 for v in evaluate(3).values())

