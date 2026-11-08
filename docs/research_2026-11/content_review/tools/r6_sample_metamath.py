import json, random, sys, collections
R='E:/MATH-REASONING-THESIS/vi-math-smt/data/test/metaMATH/metaMathQA-395K_json_vi.jsonl'
rng=random.Random(20261107); k=150; res=[]; n=0; types=collections.Counter(); bad=0
with open(R,encoding='utf-8') as f:
    for i,l in enumerate(f):
        n+=1
        if i<k: res.append((i,l))
        else:
            j=rng.randint(0,i)
            if j<k: res[j]=(i,l)
res.sort()
out=[]
for i,l in res:
    try: out.append({'index':i,'row':json.loads(l)})
    except Exception as e: out.append({'index':i,'raw':l[:2000],'err':str(e)})
json.dump(out,open(sys.argv[1]+'/work/samples_mm.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
print('lines',n)
