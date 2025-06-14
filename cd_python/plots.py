import matplotlib.pyplot as plt
import pandas as pd

def scatter(df:pd.DataFrame, title:str):
    fig,ax = plt.subplots(figsize=(5,3))
    ax.scatter(df["time"], df["mse"])
    for _,r in df.iterrows():
        ax.annotate(r["scheme"], (r["time"], r["mse"]), fontsize=6, alpha=.6)
    ax.set_xlabel("czas [s]"); ax.set_ylabel("MSE"); ax.set_title(title)
    fig.tight_layout()
    return fig
