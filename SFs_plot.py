import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import scipy
from sklearn.metrics import r2_score
import seaborn as sns
sns.set()


data = pd.read_csv('SF_HSF_COCO.csv')

def SFvsHSF(sf,hsf,num):
    
    regr = linear_model.LinearRegression()
    print(scipy.stats.pearsonr(sf,hsf))
    print(r2_score(sf,hsf))
    sf =  np.reshape(sf,((len(sf), 1)))
    hsf =  np.reshape(hsf,(len(hsf), 1))
    regr.fit(sf, hsf)
    x_test = sf
    y_pred = regr.predict(x_test)
    x = [i[0] for i in sf]
    y = [i[0] for i in hsf]
    
    fig, ax = plt.subplots(1)
    ax.scatter(sf, hsf,  color='black')
    my_data = pd.DataFrame({'x':x, 'y':y})
    sns.regplot(x='x',y='y', ci=95, data=my_data)

    #ax.plot(x_test, y_pred, color='blue', linewidth=3)
    plt.title('SF vs HSF')
    plt.xlabel('Semantic Fidelity')
    plt.ylabel('Human Semantic Fidelity ')
    plt.savefig('SF'+str(num)+'vsHSF.png')
    plt.show()
    
HSF = [value for value in data['HSF']]
SF_1 = [value for value in data['SF_1']]
SF_2 = [value for value in data['SF_2']]
SF_3 = [value for value in data['SF_3']]
SF_4 = [value for value in data['SF_4']]

SFvsHSF(SF_1,HSF,1)
SFvsHSF(SF_2,HSF,2)
SFvsHSF(SF_3,HSF,3)
SFvsHSF(SF_4,HSF,4)



