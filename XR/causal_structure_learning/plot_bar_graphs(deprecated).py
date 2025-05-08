import matplotlib.pyplot as plt
import numpy as np

birds_imp_keys = ['Size',  'Eye Color', 'Bill Shape', 'Belly Color',\
                 'Upperparts Color', 'Underparts Color', 'Back Pattern',\
                 'Bill Length', 'Undertail Color', 'Primary Color',\
                 'Wing Color', 'Bill Color', 'Breast Color', 'Back Color',\
                 'Uppertail Color', 'Throat Color', 'Belly Pattern',\
                  'Breast Pattern', 'Tail Pattern', 'Leg Color','Crown Color',\
                  'Nape Color', 'Wing Pattern','Forehead Color', 'Wing Shape',\
                  'Tail Shape']
birds_imp_keys.reverse()

#{'cardiomegaly': 0, 'atelectasis': , 'effusion': , 'infiltration': 100, 'mass': 15, 'nodule': 7, 'pneumothorax': 0, 'y_pred': 0}
birds_imp_values = [43/50, 36/50, 34/50, 31/50, 31/50, 30/50, 30/50, 30/50,\
                    29/50, 29/50,  29/50,  29/50, 29/50, 29/50, 29/50,\
                    29/50, 28/50, 28/50, 27/50, 26/50, 25/50 , 24/50, 23/50,\
                    20/50 , 17/50, 13/50]
birds_imp_values.reverse()
# {'Size': 0.825, 'Bill Length': 0.625, 'Bill Shape': 0.625, 'Eye Color': 0.575, 'Forehead Color': 0.525, 'Upperparts Color': 0.525,\
# 	     'Underparts Color': 0.525, 'Wing Shape': 0.525, 'Back Pattern': 0.525, 'Upper Tail Color': 0.525, 'Breast Pattern': 0.525,\
# 	     'Under Tail Color': 0.525, 'Tail Shape': 0.525, 'Crown Color': 0.525, 'Belly Pattern': 0.525, 'Tail Pattern': 0.525,\
# 	     'Wing Color': 0.525, 'Nape Color': 0.525, 'Wing Pattern': 0.525, 'Throat Color': 0.525, 'Back Color': 0.525,\
# 	     'Bill Color': 0.525, 'Primary Color': 0.525, 'Leg Color': 0.525, 'Belly Color': 0.525, 'Breast Color': 0.525}

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
