import os
import numpy as np
import pandas as pd
ai_df = pd.read_csv('../csv_export/ai_df.csv')
for i in range(len(ai_df)):
    if(isinstance(ai_df['twitter_url'][i], str)):
        print(ai_df['twitter_url'][i].split("/")[-1])
        os.system("python Exporter.py --query '@" + ai_df['twitter_url'][i].split("/")[-1] + "' --since 2016-01-01 --until 2018-12-31 --maxtweets 100000")
        os.rename("output_got.csv", ai_df['permalink'][i].split("/")[-1] + ".csv")

# python main.py --query '@taohq' --since 2016-01-01 --until 2018-12-31 --max-tweets 1000000
# python Exporter.py --username "taohq" --since 2016-01-01 --until 2018-12-31 --maxtweets 1000000
# python Exporter.py --query '@taohq' --since 2016-01-01 --until 2018-12-31 --maxtweets 1000000
