import pandas as pd

d = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data=d)

param = {'a':3, 'b':4, 'c':5}


def parameter_call(df, a=1, b=2, c=3):
    print(a+b+c)

parameter_call(df)
parameter_call(df, **param)
