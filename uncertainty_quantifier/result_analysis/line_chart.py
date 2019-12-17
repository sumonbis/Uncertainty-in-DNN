import random

from uncertainty_quantifier.result_analysis.utils import read_result
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def main():
    uncertain_file='../temp/overall.csv'
    rs_file="../../Random Search/result/randomSearch.csv"


    # uncertain_h,uncertain_acc=read_result(uncertain_file)
    # rs_h, rs_acc = read_result(rs_file)

    # print(uncertain_h, uncertain_acc)
    # print(rs_h, rs_acc)

    uncertain_data=pd.read_csv(uncertain_file)
    rs_data = pd.read_csv(rs_file)

    print(uncertain_data)
    print(rs_data)

    # data_joined=pd.concat([uncertain_data, rs_data])
    #
    # print(data_joined)

    sns.set_style('darkgrid')
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

    sns.lineplot(x='hidden unit size', y='accuracy', data=rs_data)
    sns.lineplot(x='hidden unit size', y='accuracy', data=uncertain_data)

    plt.legend(labels=['Random Search', 'Uncertain Random Search'])
    plt.xlabel('Hidden Unit Size')
    plt.ylabel('Accuracy')

    plt.savefig('../evaluation/line_chart.png')

    plt.show()







if __name__=="__main__":
    main()