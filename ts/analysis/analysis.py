# -*- coding: utf-8 -*-
import pandas as pd
from matplotlib import pyplot as plt


class Analysis(object):

    @staticmethod
    def polt_agg_qty():
        agg_qty_info_df = pd.read_csv("data/agg_qty_info.csv")
        agg_qty_info_df.sort_values(by=["mihome", "day"], inplace=True)

        # A bit of pre-processing to make it nicer
        agg_qty_info_df['day'] = pd.to_datetime(agg_qty_info_df['day'], format='%Y%m%d')

        for mihome in agg_qty_info_df["mihome"].drop_duplicates():
            mihome_qty_info = agg_qty_info_df[agg_qty_info_df["mihome"] == mihome][["day", "qty"]]
            mihome_qty_info.set_index(['day'], inplace=True)
            mihome_qty_info["zero"] = 0

            # Plot the data
            mihome_qty_info.plot()
            plt.ylabel('qty')
            plt.xlabel('day')
            plt.title(f"mihome_{mihome}")
            plt.savefig(f"./data/fig/agg_qty_mihome_{mihome}.jpg")
            plt.close()


if __name__ == "__main__":
    Analysis.polt_agg_qty()
