import re, pandas as pd
df = pd.read_csv('SpectroFood_dataset.csv')
df.columns.values[0] = 'label'  # first col

def parse(lbl):
    lbl = str(lbl).strip()
    m = re.match(r'(FX10_L|[ABLM])0*(\d+)', lbl)
    if not m: return (None, None)
    pre = m.group(1)
    cat = {'A':'apple','B':'broccoli','L':'leek','M':'mushroom','FX10_L':'leek'}[pre]
    return (cat, int(m.group(2)))

df[['cat','idx']] = df['label'].apply(lambda x: pd.Series(parse(x)))
print(df.groupby('cat').size())   # confirm per-category counts

print(df.groupby('cat')['DRY MATTER'].describe())