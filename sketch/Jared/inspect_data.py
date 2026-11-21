from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download

DATA = Path(__file__).resolve().parent.parent / 'data'   # repo root data/
DATA.mkdir(exist_ok=True)
print('Downloading official SentinelDesk dataset...')
path = hf_hub_download(
    repo_id='Tobi-Bueck/customer-support-tickets',
    repo_type='dataset',
    filename='aa_dataset-tickets-multi-lang-5-2-50-version.csv',
)
df = pd.read_csv(path)
print('\n===== DATASET INFO =====')
print('Total rows:', len(df))
print('Columns:', df.columns.tolist())
print('\n===== FIRST 3 TICKETS =====')
print(df[['subject', 'body', 'type', 'queue', 'priority']].head(3).to_string())
print('\n===== MISSING VALUES =====')
print(df[['subject', 'body', 'type', 'queue', 'priority']].isna().sum())
print('\n===== LABEL DISTRIBUTIONS =====')
for column in ['type', 'queue', 'priority', 'language']:
    print(f'\n{column}:')
    print(df[column].value_counts(dropna=False).to_string())
df.to_csv(DATA / 'raw_tickets.csv', index=False)
print('\nSUCCESS: Dataset saved to', DATA / 'raw_tickets.csv')
