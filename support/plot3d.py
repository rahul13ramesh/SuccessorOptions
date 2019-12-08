from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot3d(mat, path):
    """
    Plots a matrix of data-points in 3-D
    """
    data = []
    for r, row in enumerate(mat):
        for c, val in enumerate(row):
            data.append([r, c, val])

    df = pd.DataFrame(data)

    # Transform it to a long format
    df.columns = ["X", "Y", "Z"]

    # And transform the old column name in something numeric
    #  df['X']=pd.Categorical(df['X'])
    #  df['X']=df['X'].cat.codes

    # Make the plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(df['Y'], df['X'], df['Z'],
                    cmap=plt.cm.viridis, linewidth=0.2)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, ax.zaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
        item.set_fontsize(22)
    ax.tick_params(axis='both', which='major', pad=-1)

    plt.xticks([0, 5, 10])
    plt.yticks([0, 5, 10])
    ax.set_zticks([0, 6, 12])

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close()

