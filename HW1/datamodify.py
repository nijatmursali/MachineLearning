import pandas as pd


data = pd.read_csv("turboaz.csv")

yurush = data['Yurush']
qiymet = data['Qiymet']
buraxilishili = data['Buraxilish ili']

for i in range(0, 1328):
    if ' ' in yurush[i]:
        yurush[i] = (yurush[i].replace(' ',''))
    if 'km' in yurush[i]:
        yurush[i] = float(yurush[i].replace('km',''))

for i in range(0,1328):
    if 'AZN' in qiymet[i]:
        qiymet[i] = (qiymet[i].replace('AZN',''))
    if '$' in qiymet[i]:
        qiymet[i] = float(qiymet[i].replace('$',''))
        qiymet[i] = qiymet[i] * 1.7




data.to_csv("turboazmodified.csv")
