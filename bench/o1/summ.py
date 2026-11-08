import json,glob,sys
for f in sorted(glob.glob(sys.argv[1])):
    try: r=json.load(open(f))
    except Exception as e: print(f,'ERR',e); continue
    if 'runs' in r or r.get('mode'):
        print(f, r.get('ok'), (r.get('error') or '')[:300], [{k:v for k,v in x.items() if 'text' not in k} for x in r.get('runs',[])])
    else:
        print(f, {k:r.get(k) for k in ('ok','tok_per_s','s_per_step','peak_mem_gb','pad_frac','packing_correct','logit_check_max_nll_diff','load_s','error')})
