"""Compute aggregated relative scores from Chronos benchmark CSVs.

Usage: python compute_agg_scores.py
"""
import csv, math, glob, os

base = os.path.dirname(os.path.abspath(__file__))

# Load seasonal naive baseline
sn = {}
with open(os.path.join(base, 'seasonal-naive-zero-shot.csv')) as f:
    for row in csv.DictReader(f):
        sn[row['dataset']] = {'MASE': float(row['MASE']), 'WQL': float(row['WQL'])}

# Process each model
models = sorted(glob.glob(os.path.join(base, 'chronos-*-zero-shot.csv')))
results = []
for mf in models:
    model_name = os.path.basename(mf).replace('-zero-shot.csv', '')
    data = {}
    with open(mf) as f:
        for row in csv.DictReader(f):
            data[row['dataset']] = {'MASE': float(row['MASE']), 'WQL': float(row['WQL'])}
    # Compute relative scores (model / seasonal_naive) then geometric mean
    rel_mase = []
    rel_wql = []
    for ds in sn:
        if ds in data:
            rel_mase.append(data[ds]['MASE'] / sn[ds]['MASE'])
            rel_wql.append(data[ds]['WQL'] / sn[ds]['WQL'])
    geo_mase = math.exp(sum(math.log(x) for x in rel_mase) / len(rel_mase))
    geo_wql = math.exp(sum(math.log(x) for x in rel_wql) / len(rel_wql))
    results.append((model_name, geo_mase, geo_wql, len(rel_mase)))

print(f"{'Model':<30} {'Agg Rel MASE':>14} {'Agg Rel WQL':>14} {'#DS':>5}")
print('-' * 65)
for name, mase, wql, n in sorted(results, key=lambda x: x[1]):
    print(f"{name:<30} {mase:>14.4f} {wql:>14.4f} {n:>5}")

print("\n\nVerification against agg-rel-scores CSVs:")
print('-' * 65)
for mf in sorted(glob.glob(os.path.join(base, 'chronos-*-agg-rel-scores.csv'))):
    model_name = os.path.basename(mf).replace('-agg-rel-scores.csv', '')
    with open(mf) as f:
        content = f.read()
        if '404' in content:
            print(f"  {model_name}: file not found (404)")
            continue
        f.seek(0)
        for row in csv.DictReader(f):
            if row['benchmark'] == 'zero-shot':
                print(f"  {model_name} {row['metric']}: {float(row['value']):.4f}")
