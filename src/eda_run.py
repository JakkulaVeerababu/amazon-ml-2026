import pandas as pd
import numpy as np

base = r'c:\Users\LENOVO\Desktop\Cognate\student_resource\dataset'

s1 = pd.read_csv(f'{base}/train/train_source1.tsv', sep='\t', dtype=str).fillna('')
s2 = pd.read_csv(f'{base}/train/train_source2.tsv', sep='\t', dtype=str).fillna('')
s3 = pd.read_csv(f'{base}/train/train_source3.tsv', sep='\t', dtype=str).fillna('')
gt = pd.read_csv(f'{base}/train/train_ground_truth.tsv', sep='\t', dtype=str).fillna('')

print('=== SHAPES ===')
print(f'S1: {s1.shape}, S2: {s2.shape}, S3: {s3.shape}, GT: {gt.shape}')
print()
print('=== COLUMNS ===')
print('S1:', s1.columns.tolist())
print('GT:', gt.columns.tolist())
print()
print('=== SAMPLE S1 ===')
print(s1.head(3).to_string())
print()
print('=== COUNTRY DIST ===')
for name, df in [('S1', s1), ('S2', s2), ('S3', s3)]:
    print(f'  {name}:', df['country'].value_counts().to_dict())

print()
print('=== GT STATS ===')
gt['n_matches'] = gt['matched_entity_ids'].apply(lambda x: len(x.split(',')) if x.strip() else 0)
n_sing = (gt['n_matches']==0).sum()
n_match = (gt['n_matches']>0).sum()
print('Singletons (0):', n_sing, round(100*n_sing/len(gt), 1), '%')
print('Has matches:', n_match, round(100*n_match/len(gt), 1), '%')
print('Max matches:', gt['n_matches'].max())
print('Avg (non-singleton):', round(gt[gt['n_matches']>0]['n_matches'].mean(), 2))
print('Match distribution:')
print(gt['n_matches'].value_counts().sort_index().head(15))
print()
print('=== SAMPLE GT ===')
print(gt[gt['n_matches']>0].head(5).to_string())
