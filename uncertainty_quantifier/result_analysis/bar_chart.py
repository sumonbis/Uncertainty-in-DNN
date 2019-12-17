
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def get_counts(ranges, vals):

    result=[0]*(len(ranges)-1)

    for i in range(len(vals)):
        cnt=0
        for j in range(1,len(ranges)):

            if vals[i]>=ranges[j-1] and vals[i]<ranges[j]:
                result[j-1]=result[j-1]+1
            if j==len(ranges)-1 and vals[i]==ranges[j]:
                result[j - 1] = result[j - 1] + 1
    return result

def main():
    uncertain_file='../temp/overall.csv'
    rs_file="../../Random Search/result/randomSearch.csv"


    uncertain_data=pd.read_csv(uncertain_file)
    rs_data = pd.read_csv(rs_file)

    print(uncertain_data)
    print(rs_data)

    mx=max(uncertain_data['accuracy'].max(), rs_data['accuracy'].max())
    mn = min(uncertain_data['accuracy'].min(), rs_data['accuracy'].min())
    diff=mx-mn
    grp=5

    ranges=np.linspace(mn,mx,grp+1)

    print(mn, mx)
    print(ranges)
    print(ranges[len(ranges)-1]>=mx)

    uncertain_counts=get_counts(ranges, list(uncertain_data['accuracy']))
    rs_counts=get_counts(ranges, list(rs_data['accuracy']))
    x_labels=[]
    for i in range(1,len(ranges)):
        x_labels.append(str(round(100*ranges[i-1],2))+' - '+str(round(100*ranges[i],2)))

    print(x_labels)

    print(get_counts(ranges, list(uncertain_data['accuracy'])))
    print(get_counts(ranges, list(rs_data['accuracy'])))

    uncertain_data=pd.DataFrame({'accuracy range': x_labels, 'Uncertain Random Search': uncertain_counts, 'Random Search': rs_counts})
    uncertain_data=pd.melt(uncertain_data, id_vars=['accuracy range'], value_vars=['Uncertain Random Search', 'Random Search'], var_name='Type')
    # rs_data=pd.DataFrame({'accuracy range': x_labels, 'count': rs_counts})

    print(uncertain_data)
    sns.set_style('darkgrid')
    sns.barplot(x='accuracy range',y='value',hue='Type',data=uncertain_data)

    plt.xlabel('Accuracy Range')
    plt.ylabel('Count')

    plt.savefig('../evaluation/bar_chart.png')
    plt.show()









if __name__=="__main__":
    main()