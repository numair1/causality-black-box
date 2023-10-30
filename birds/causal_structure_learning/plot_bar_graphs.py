import matplotlib.pyplot as plt
import numpy as np

x  = {'Belly Color': 2/50., 'Belly Pattern': 0/50., 'Forehead Color': 1/50., 'Wing Color': 0/50., 'Size': 9/50., 'Wing Shape': 45/50., 'Wing Pattern': 39/50., 'Crown Color': 0/50., 'Back Color': 0/50., 'Primary Color': 4/50., 'Leg Color': 3/50., 'Upper Tail Color': 13/50., 'Throat Color': 20/50., 'Nape Color': 5/50., 'Bill Color': 0/50., 'Bill Shape': 0/50., 'Under Tail Color': 6/50., 'Eye Color': 0/50., 'Back Pattern': 0/50., 'Breast Color': 6/50., 'Upperparts Color': 9/50., 'Tail Shape': 36/50., 'Underparts Color': 11/50., 'Tail Pattern': 1/50., 'Breast Pattern': 1/50., 'Bill Length': 0/50.}

birds_dict = dict(sorted(x.items(), key=lambda item: item[1]))
#{'cardiomegaly': 0, 'atelectasis': , 'effusion': , 'infiltration': 100, 'mass': 15, 'nodule': 7, 'pneumothorax': 0, 'y_pred': 0}
birds_imp_keys = list(birds_dict.keys())

birds_imp_values = list(birds_dict.values())

# {'Size': 0.825, 'Bill Length': 0.625, 'Bill Shape': 0.625, 'Eye Color': 0.575, 'Forehead Color': 0.525, 'Upperparts Color': 0.525,\
# 	     'Underparts Color': 0.525, 'Wing Shape': 0.525, 'Back Pattern': 0.525, 'Upper Tail Color': 0.525, 'Breast Pattern': 0.525,\
# 	     'Under Tail Color': 0.525, 'Tail Shape': 0.525, 'Crown Color': 0.525, 'Belly Pattern': 0.525, 'Tail Pattern': 0.525,\
# 	     'Wing Color': 0.525, 'Nape Color': 0.525, 'Wing Pattern': 0.525, 'Throat Color': 0.525, 'Back Color': 0.525,\
# 	     'Bill Color': 0.525, 'Primary Color': 0.525, 'Leg Color': 0.

XR_imp_keys = ['Infiltration', 'Atelectasis', 'Effusion', 'Mass', 'Nodule', 'Cardiomegaly', 'Pneumothorax']
XR_imp_values = [1, 0.97 , 0.93, 0.15, 0.07, 0, 0]
XR_imp_keys.reverse()
XR_imp_values.reverse()

colors = np.random.rand(26,3)
h = plt.barh(np.arange(1, len(birds_imp_keys)+ 1), birds_imp_values, color = 'darkred')
plt.yticks(ticks = np.arange(1, len(birds_imp_keys) + 1), labels = birds_imp_keys, fontsize = 12, fontweight = 'bold')
plt.xlabel("Edge Stability", fontsize = 14, fontweight = 'bold')
plt.title("Bird Classification Experiment", fontsize = 14, fontweight = 'bold')
#plt.legend(h, birds_imp_keys)
plt.show()

plt.clf()

plt.barh(np.arange(1, len(XR_imp_keys)+ 1), XR_imp_values, color = 'darkred')
plt.yticks(ticks = np.arange(1, len(XR_imp_keys) + 1), labels = XR_imp_keys, fontsize = 14, fontweight = 'bold')
plt.xlabel("Edge Stability", fontsize = 14, fontweight = 'bold')
plt.title("X-ray Classification Experiment", fontsize = 14, fontweight = 'bold')
plt.show()
