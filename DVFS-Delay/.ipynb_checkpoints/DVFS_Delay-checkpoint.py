import pandas as pd
from pathlib import Path

FreqMeasurementsFile=Path("./Data/FreqMeasurements.csv").resolve()
Datacsv=Path("./Data/Data.csv").resolve()

with open(FreqMeasurementsFile) as f:
    df=pd.read_csv(f)
#df['TimeFreq2']=df["TimeTotal"] - df['TimeFreq1']
#df['TimeFreq2'] = df.apply(lambda x: x['TimeFreq1'] if x['TimeFreq2'] < 0 else x['TimeFreq2'], axis=1)
#df['TimeFreq2'] = df.apply(lambda x: x['TimeFreq1'] if x['TimeFreq2'] < 0 else x['TimeTotal'] - x['TimeFreq1'], axis=1)
#df['TimeFreq2'] = np.where(df['TimeFreq2'] < 0, df['TimeFreq1'], df[' TimeTotal'] - df[' TimeFreq1'])


df.groupby(["Num_iterations","PE","Freq","NextFreq"]).sum(['AVG']).reset_index()


# +
def Calc_Delay(freq1=3, freq2=5, PE="Little", num_iterations=20000000):
    if freq1==freq2:
        return 0
    T1=df[(df['Num_iterations']==num_iterations) & (df['PE']==PE) & (df['Freq']==freq1) &(df['NextFreq']==freq1)]["AVG"].iloc[0]
    T2=df[(df['Num_iterations']==num_iterations) & (df['PE']==PE) & (df['Freq']==freq2) &(df['NextFreq']==freq2)]["AVG"].iloc[0]
    T1_2=df[(df['Num_iterations']==num_iterations) & (df['PE']==PE) & (df['Freq']==freq1) &(df['NextFreq']==freq2)]["AVG"].iloc[0]
    Delay=T1*(T1_2-T2)/(T1-T2)
    print(T1,T2,T1_2)
    print(f'Delay is {Delay}')
    return Delay

def Cal_Delay(row):
    freq1=row['Freq'].iloc[0]
    freq2=row['NextFreq'].iloc[0]
    PE=row['PE'].iloc[0]
    num_iterations=row['Num_iterations'].iloc[0]
    T1=df[(df['Num_iterations']==num_iterations) & (df['PE']==PE) & (df['Freq']==freq1) &(df['NextFreq']==freq1)]["AVG"].iloc[0]
    T2=df[(df['Num_iterations']==num_iterations) & (df['PE']==PE) & (df['Freq']==freq2) &(df['NextFreq']==freq2)]["AVG"].iloc[0]
    T1_2=df[(df['Num_iterations']==num_iterations) & (df['PE']==PE) & (df['Freq']==freq1) &(df['NextFreq']==freq2)]["AVG"].iloc[0]
    Delay=T1*(T1_2-T2)/(T1-T2)
    print(T1,T2,T1_2)
    print(f'Delay is {Delay}')
    return Delay



# -

df['Delay'] = df.apply(lambda x: Calc_Delay(x['Freq'], x['NextFreq'], x['PE'], x['Num_iterations']), axis=1)
#df['Delay'] = df.apply(Cal_Delay, axis=1)
display(df)
df.to_csv(Datacsv)

pivoted = df.pivot_table(index=['Num_iterations','PE'], columns=['Freq','NextFreq'], values=['AVG'])
print(pivoted)


fig, axs = plt.subplots(ncols=len(voltages), figsize=(12, 4))
for i, voltage in enumerate(voltages):
    ax = axs[i]
    df[df['freq'] == freq].plot(x='time', y='voltage', ax=ax, label=f'voltage={voltage} V')
    ax.set_title(f'Voltage vs. Time at {freq} Hz and {voltage} V')
plt.tight_layout()
plt.show()

df


