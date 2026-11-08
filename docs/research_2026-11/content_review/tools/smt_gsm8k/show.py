import json,random,sys
SP=sys.argv[1]; a,b=int(sys.argv[2]),int(sys.argv[3])
F={}
for l in open(SP+'/features.jsonl',encoding='utf-8'):
    x=json.loads(l); F[x['index']]=x
R={json.loads(l)['index']:json.loads(l) for l in open('data/formalize_output/gsm8k/gsm8k-smt-nvidia-FULL.jsonl',encoding='utf-8')}
S=sorted(random.Random(20261107).sample(range(7473),150))
if a==0 and b==0: print(S); sys.exit()
for k,i in enumerate(S[a:b],start=a):
    f=F[i]; r=R[i]
    smt='\n'.join(l for l in r['smt_lib'].splitlines() if not l.startswith(('(set-logic','(check-sat','(get-value','(get-model')))
    print(f"#{k} idx={i} gold={f['gold']} z3={f.get('value')} st={f['status']} match={f.get('match')} cls={f.get('cls')}:{f.get('why')} uniq={f.get('unique_wo_anslit')} SL={f.get('straight_line')} ret={r.get('retries')} unexpl={f.get('unexplained_lits')}")
    print("Q:",r['question'])
    print(smt); print()
