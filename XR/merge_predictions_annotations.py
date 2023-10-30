import pandas as pd
import numpy as np

csv = pd.read_csv("Data_Entry_2017_v2020.csv")
y_pred = pd.read_csv("./data_alternate_baseline/model_preds_normal_baseline.txt", names = ['Image Index', 'y_pred'])

df = csv.merge(y_pred, on = 'Image Index', how = 'inner')
df = df[['Image Index', 'Finding Labels', 'y_pred']]

cols = ['y_pred', 'cardiomegaly', 'atelectasis', 'effusion', \
            'infiltration', 'mass', 'nodule', 'pneumothorax']

labels_df = pd.DataFrame(columns = cols)
print(labels_df)
for index, row in df.iterrows():
     row_dict = {'y_pred': 0, 'cardiomegaly': 0, 'atelectasis': 0, 'effusion': 0, \
                 'infiltration':0 , 'mass':0, 'nodule':0, 'pneumothorax': 0}
     row_findings = row['Finding Labels'].split("|")

     row_dict['y_pred'] = row.y_pred
     row_dict['cardiomegaly'] = int('Cardiomegaly' in row_findings)
     row_dict['atelectasis'] = int('Atelectasis' in row_findings)
     row_dict['effusion'] = int('Effusion' in row_findings)
     row_dict['infiltration'] = int('Infiltration' in row_findings)
     row_dict['mass'] = int('Mass' in row_findings)
     row_dict['nodule'] = int('Nodule' in row_findings)
     row_dict['pneumothorax'] = int('Pneumathorax' in row_findings)

     labels_df = labels_df.append(row_dict, ignore_index = True)

labels_df.to_csv("./data_alternate_baseline/combined_pred_annotated.csv", index = False)
