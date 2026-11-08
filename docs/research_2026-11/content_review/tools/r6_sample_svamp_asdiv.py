import json, random, re, sys
import xml.etree.ElementTree as ET
R='E:/MATH-REASONING-THESIS/vi-math-smt/'
S=sys.argv[1]
out={}
# SVAMP
vi=[json.loads(l) for l in open(R+'data/translation_test_output/svamp/svamp-vi.jsonl',encoding='utf-8')]
en={d['ID']:d for d in json.load(open(S+'/dl_svamp/SVAMP.json',encoding='utf-8'))}
print('svamp vi',len(vi),'en',len(en), 'ids matched', sum(v['ID'] in en for v in vi))
idx=sorted(random.Random(20261107).sample(range(len(vi)),150))
out['vi_svamp']=[{'index':i,'vi':vi[i],'en':en.get(vi[i]['ID'])} for i in idx]
# ASDiv
vi=[json.loads(l) for l in open(R+'data/translation_test_output/asdiv/asdiv-validation-vi.jsonl',encoding='utf-8')]
root=ET.parse(S+'/dl_asdiv/ASDiv.xml').getroot()
en=[]
for p in root.iter('Problem'):
    en.append({'ID':p.get('ID'),'Grade':p.get('Grade'),'body':p.findtext('Body'),'question':p.findtext('Question'),'answer':p.findtext('Answer'),'type':p.findtext('Solution-Type'),'formula':p.findtext('Formula')})
print('asdiv vi',len(vi),'en',len(en))
mis=[i for i in range(min(len(vi),len(en))) if vi[i]['formula']!=en[i]['formula']]
print('formula mismatches by position',len(mis),mis[:10])
idx=sorted(random.Random(20261107).sample(range(len(vi)),150))
out['vi_asdiv']=[{'index':i,'vi':vi[i],'en':en[i]} for i in idx]
json.dump(out,open(S+'/work/samples_sv_as.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
