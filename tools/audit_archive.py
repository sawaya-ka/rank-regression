"""Read-only consistency audit; does not rerun numerical experiments."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import csv, hashlib, json, math
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--extracted', type=Path, required=True, help='Root containing results/ and source/ from the reviewed raw archive')
parser.add_argument('--output-dir', type=Path, default=Path('build/archive-audit'))
args = parser.parse_args()
ROOT = args.extracted
OUT = args.output_dir
R = ROOT / 'results'
def read(p): return json.loads(p.read_text())
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def key(row): return (round(float(row['df']), 10), round(float(row['delta']), 10), row.get('penalty', ''))
def equivalent(a, b):
    if a is None: return b in ('', 'nan', 'NaN', 'None')
    if isinstance(a, bool): return str(a).lower() == str(b).lower()
    if isinstance(a, (float, int)):
        try: return math.isclose(a, float(b), rel_tol=1e-11, abs_tol=1e-13)
        except ValueError: return False
    return str(a) == str(b)

report = {'scope': 'Hashes, checkpoint numerical gates, table reconciliation, and asset completeness; no expensive experiments rerun.', 'errors': [], 'warnings': []}
errors = report['errors']
manifest = read(R/'repair_manifest.json')
prov = read(R/'checkpoint_provenance.json')
parents = {x['job_id']: x for x in manifest['parent_checkpoints']}
provenance = {x['job_id']: x for x in prov['checkpoints']}
files = sorted((R/'checkpoints').glob('*.json'))
if len(files) != len(provenance) or len(parents) != len(provenance): errors.append('Checkpoint count mismatch')
datasets = read(R/'data/recomputation_manifest.json')['datasets']
report['dataset_hashes'] = {}
tables = {}
for name, item in datasets.items():
    actual = digest(R/'data'/name)
    report['dataset_hashes'][name] = {'sha256': actual, 'matches_manifest': actual == item['sha256']}
    if actual != item['sha256']: errors.append('Dataset hash mismatch: '+name)
    with (R/'data'/name).open() as f: rows = list(csv.DictReader(f))
    entry = {'rows': len(rows)}
    if rows and 'df' in rows[0] and 'delta' in rows[0]:
        tables[name] = {key(row): row for row in rows}
        entry['unique_coordinate_count'] = len(tables[name])
        if len(tables[name]) != len(rows): errors.append('Duplicate coordinates: '+name)
    report['dataset_hashes'][name].update(entry)
counts, origins, raw_counts, statuses = Counter(), Counter(), Counter(), Counter()
hash_verified = 0
field_comparisons = 0
infinite_ls_serialization_conversions = 0
missing_update_columns = defaultdict(set)
maxima = defaultdict(float)
orders = Counter()
repaired = []
raw_by_estimator = Counter()
trace_invalid = Counter()
map_table = {'quadrature':'unpenalized_student_t.csv', 'theory_lasso':'theory/penalized_phase.csv', 'unpenalized_rank':'empirical/unpenalized_phase.csv', 'penalized_repair':'empirical/penalized_phase.csv'}
for path in files:
    data = path.read_bytes(); sha = hashlib.sha256(data).hexdigest(); c = json.loads(data)
    jid = path.stem; kind = c['kind']; counts[kind] += 1
    p = provenance[jid]; origins[p['origin']] += 1
    if sha != p['sha256'] or c.get('job_id') != jid or not c.get('complete'): errors.append('Checkpoint verification: '+jid)
    else: hash_verified += 1
    if p['origin'] == 'parent':
        if sha != parents[jid]['sha256'] or not parents[jid]['complete'] or c['fingerprint'] != manifest['parent_fingerprint']: errors.append('Parent provenance: '+jid)
    else:
        if c['fingerprint'] != manifest['fingerprint']: errors.append('Repair fingerprint: '+jid)
        parent_file = R/'parent_failures'/path.name
        old = read(parent_file)
        if digest(parent_file) != parents[jid]['sha256'] or old['complete'] or c['repair']['parent_checkpoint_sha256'] != parents[jid]['sha256']: errors.append('Retained failure provenance: '+jid)
        repaired.append({'job_id': jid, 'kind': kind, 'elapsed_seconds': c['elapsed_seconds']})
        if kind == 'quadrature':
            if c['trace'][:len(old['trace'])] != old['trace']: errors.append('Quadrature history altered: '+jid)
        if kind == 'unpenalized_rank':
            changed = [i for i,(a,b) in enumerate(zip(old['raw'],c['raw'])) if a != b]
            if changed != [4] or len(old['raw']) != len(c['raw']) or c['trace'][0]['prior_record'] != old['raw'][4]: errors.append('Empirical repair not confined to rep 4')
            report['fixed_rho_repair'] = {'job_id':jid,'changed_repetitions':changed,'preserved_repetitions':len(c['raw'])-len(changed),'seed':c['raw'][4]['seed'],'iterations':c['raw'][4]['solver_iterations'],'rho':c['raw'][4]['solver_rho'],'mse':c['raw'][4]['mse_rank']}
    u = c['updates']; table_name = map_table[kind]; tablerow = tables[table_name].get(key(u))
    if tablerow is None: errors.append('Missing table coordinate: '+jid)
    else:
        for field, value in u.items():
            if field not in tablerow: missing_update_columns[table_name].add(field); continue
            field_comparisons += 1
            # The strict-JSON checkpoint cleaner maps infinite risks to null;
            # the CSV merger restores +inf for LS at df <= 2 explicitly.
            if kind == 'quadrature' and u['df'] <= 2 and field in ('alpha2_LS','alpha_LS') and value is None and tablerow[field] == 'inf':
                infinite_ls_serialization_conversions += 1
                continue
            if not equivalent(value, tablerow[field]): errors.append(f'Table mismatch: {jid} {field} {value!r} != {tablerow[field]!r}')
    if kind == 'quadrature':
        last, prev = c['trace'][-1], c['trace'][-2]
        delta = max(abs(last[k]-prev[k])/last[k] for k in ('rank','lad'))
        if delta > 5e-4 or not math.isclose(delta,last['max_relative_change'],rel_tol=1e-13): errors.append('Quadrature gate: '+jid)
        if any(not (math.isfinite(last[k]) and last[k] > 0) for k in ('rank','lad')): errors.append('Quadrature positivity: '+jid)
        maxima['quadrature_relative_change'] = max(maxima['quadrature_relative_change'],delta)
        orders[last['order']] += 1
        if p['origin'] == 'repaired': repaired[-1].update(order=last['order'],max_relative_change=delta)
    if kind == 'theory_lasso':
        for est in ('ls','lad','rank'):
            if est == 'ls' and u['df'] <= 2:
                if u[f'{est}_status'] != 'infinite_error_variance': errors.append('LS infinite variance gate: '+jid)
                continue
            v,cost = u[f'alpha2_{est}_opt'],u[f'{est}_residual_squared']
            if not(math.isfinite(v) and v>0 and math.isfinite(cost) and cost<=1e-5): errors.append('Theory gate: '+jid+' '+est)
            maxima['theory_selected_residual_squared'] = max(maxima['theory_selected_residual_squared'],cost)
    for t in c.get('trace',[]):
        if t.get('valid') is False: trace_invalid[kind] += 1
    for raw in c.get('raw',[]):
        raw_counts[kind] += 1; raw_by_estimator[kind+':'+raw.get('estimator','rank')] += 1
        if kind == 'unpenalized_rank': success = raw.get('rank_success'); status=raw.get('solver_status')
        else: success = raw.get('success'); status=raw.get('solver_termination',raw.get('message'))
        statuses[kind+':'+str(status)] += 1
        if not success: errors.append('Unsuccessful selected repetition: '+jid+' '+str(raw['rep']))
        if raw.get('estimator','rank') == 'rank':
            for metric, tol in [('duality_gap','gap_tolerance'),('primal_residual','primal_tolerance'),('dual_residual','dual_tolerance')]:
                v,t = raw.get('solver_'+metric), raw.get('solver_'+tol)
                if v is None or t is None or not(math.isfinite(v) and math.isfinite(t) and v<=t*(1+1e-12)):
                    errors.append('Rank numerical gate: '+jid+' '+str(raw['rep'])+' '+metric)
                elif t>0: maxima[kind+'_'+metric+'_tolerance_ratio'] = max(maxima[kind+'_'+metric+'_tolerance_ratio'],v/t)
report.update(checkpoint_counts=dict(counts),checkpoint_origins=dict(origins),checkpoint_hashes_verified=hash_verified,raw_repetitions=dict(raw_counts),raw_by_estimator=dict(raw_by_estimator),raw_solver_statuses=dict(statuses),maxima=dict(maxima),quadrature_orders=dict(orders),repaired_jobs=repaired,table_field_comparisons=field_comparisons,update_fields_not_in_csv={k:sorted(v) for k,v in missing_update_columns.items()},invalid_theory_search_candidates=dict(trace_invalid))
report['infinite_ls_serialization_conversions'] = infinite_ls_serialization_conversions
report['repair_source_hashes'] = {name:{'sha256':digest(ROOT/'source'/name),'matches_manifest':digest(ROOT/'source'/name)==sha} for name,sha in manifest['repair_source_sha256'].items()}
for name, check in report['repair_source_hashes'].items():
    if not check['matches_manifest']: errors.append('Repair source hash mismatch: '+name)
report['figures'] = [{'name':p.name,'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted((R/'figures').glob('*'))]
report['run_status'] = read(R/'run_status.json')
report['warnings'] = [
    'Finite lambda searches and quadrature-order comparisons are numerical diagnostics, not proofs of unrestricted global optima or rigorous quadrature error bounds.',
    'Only selected-lambda empirical repetitions are retained; unsuccessful lambda candidates in search traces need not invalidate an accepted selected candidate.',
    'Figure 8 is unchanged historical finite-grid data; its LS ridge optimum at df=3, delta=0.1 reaches the upper search bound.',
    'Retained LS/LAD empirical summaries use historical diagnostics except at 760 forced-signal points where all estimators were retuned.',
    'Figure 10 tunes lambda using median MSE, then displays mean MSE at that selection on the same repetitions (oracle Monte Carlo comparison, not cross-validation).',
    'Some metadata fields about replicate-level availability are stale: new raw selected repetitions exist for all Rank points and all methods at forced-signal correction points; historical retained LS/LAD raw availability must be stated separately.',
    'Archive contains generated Figures 1–10 and S1–S2. The separate S3 schematic is a reconstruction and is not in this run archive.'
]
expected_counts = {'quadrature':865, 'theory_lasso':10000, 'unpenalized_rank':2500, 'penalized_repair':5000}
if dict(counts) != expected_counts: errors.append('Unexpected production job counts')
if not report['run_status'].get('complete') or report['run_status'].get('failed_jobs') != 0: errors.append('Run is not complete')
if manifest['fingerprint'] != '80e2d1b5014ebb2019f5dd5851c53f9543b1639a00d7a4a410e26ce8edad9375': errors.append('Unexpected reviewed run fingerprint')
report['passed'] = not errors
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'received-run-audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('figures','repaired_jobs','run_status','dataset_hashes')},indent=2,ensure_ascii=False))

sys.exit(0 if report["passed"] else 1)
