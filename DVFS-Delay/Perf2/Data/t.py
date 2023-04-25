import pandas as pd

data=pd.read_csv("FreqMeasurements2_5.csv")


data.columns

# +
PE="Big"
Freq=0
NextFreq=0
condition=(data['Num_iterations']==256) &\
            (data['PE']=="Big") &\
            (data["Freq"].isin(range(1,2)))&\
            (data["NextFreq"].isin(range(0,10)))
           
display(data[condition])
data[condition]["AVG"]/1000
# -

pd.pivot_table(data,values="AVG", index=["PE","Freq"], columns=["NextFreq"])

gdata=data[data["PE"]=="Big"]
table=pd.pivot_table(gdata,values="AVG", index=["NextFreq"], columns=["Freq"])
table

table.plot()

F1=0
F2=4
data[(data["PE"]=='GPU') & (data['Freq']==F1) & (data['NextFreq']==F2)]['AVG'].mean()


