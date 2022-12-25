import os

import numpy as np
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.charts import Graph

from processing.DataPrepare import cal_similarity_net
from tools.PathTools import ChartPathGetter, TempPathGetter


def draw_graphic(cancer: str, threshold: float, method):
    dpg = TempPathGetter()
    node_path = dpg.get_path(cancer, method, 'nodes.csv')
    edge_path = dpg.get_path(cancer, method, 'edges.csv')
    if not (os.path.exists(node_path) and os.path.exists(edge_path)):
        cal_similarity_net(cancer, method)
    nodes_df: pd.DataFrame = pd.read_csv(node_path, index_col=0)
    nodes = []
    categories = []
    for c in nodes_df['stage'].drop_duplicates():
        categories.append({'name': c})
    for i, n in nodes_df.iterrows():
        nodes.append({
            'id': i,
            "name": n['entity_submitter_id'],
            'value': n['flag'],
            "category": n['stage'],
            "symbolSize": 5
        })
    links_df = pd.read_csv(edge_path, index_col=0)
    links = []
    '''
    相关系数  
    0.8-1.0     极强相关
    0.6-0.8     强相关
    0.4-0.6     中等程度相关
    0.2-0.4     弱相关
    0.0-0.2     极弱相关或无相关
    '''
    links_df = links_df[links_df['spearman'] >= threshold]
    for i, l in links_df.iterrows():
        links.append({
            # "source": barcodes[int(l.loc['i'])],
            # "target": barcodes[int(l.loc['j'])]
            "source": int(l.loc['i']),
            "target": int(l.loc['j'])
        })
        nodes[int(l.loc['i'])]["symbolSize"] += 0.5
        nodes[int(l.loc['j'])]["symbolSize"] += 0.5

    Graph(init_opts=opts.InitOpts(width="100%", height="900px")).add(
        "",
        nodes,
        links,
        categories,
        layout="circular",
        # is_rotate_label=True,
        repulsion=50,
        linestyle_opts=opts.LineStyleOpts(curve=0.2),
        label_opts=opts.LabelOpts(is_show=False),
    ).set_global_opts(
        # legend_opts=opts.LegendOpts(is_show=False),
        title_opts=opts.TitleOpts(title="Graph-" + cancer),
    ).render(ChartPathGetter().get_path(cancer, method, 'graphic.html'))


def get_similarity_bucket(cancer, method):
    dpg = TempPathGetter()
    node_path = dpg.get_path(cancer, method, 'nodes.csv')
    edge_path = dpg.get_path(cancer, method, 'edges.csv')
    if not (os.path.exists(node_path) and os.path.exists(edge_path)):
        cal_similarity_net(cancer, method)
    links_df = pd.read_csv(edge_path, index_col=0)
    '''
    相关系数  
    0.8-1.0     极强相关
    0.6-0.8     强相关
    0.4-0.6     中等程度相关
    0.2-0.4     弱相关
    0.0-0.2     极弱相关或无相关
    '''
    ranges = range(101)
    ys1 = [0 for _ in ranges]
    ys2 = [0 for _ in ranges]
    ys3 = [0 for _ in ranges]
    ys4 = [0 for _ in ranges]
    ys5 = [0 for _ in ranges]
    ys6 = [0 for _ in ranges]
    for _, l in links_df.iterrows():
        ys1[int(round(l.loc["spearman"] * 100))] += 1
        ys2[int(round(l.loc["pearson"] * 100))] += 1
        ys3[int(round(l.loc["kendall"] * 100))] += 1
        ys4[int(round(l.loc["euclidean"] * 100))] += 1
        ys5[int(round(l.loc["manhattan"] * 100))] += 1
        ys6[int(round(l.loc["chebyshev"] * 100))] += 1
    return ys1, ys2, ys3, ys4, ys5, ys6


def cal_threshold(cancer, method, percent=0.85):
    y = get_similarity_bucket(cancer, method)[0]
    total = 0
    for i in range(101):
        total += y[i]
        if total >= sum(y) * percent:
            break
    return i / 100


def draw_histogram(cancer, method):
    ys1, ys2, ys3, ys4, ys5, ys6 = get_similarity_bucket(cancer, method)
    ranges = range(101)
    xs = [(i / 100) for i in ranges]
    c = (
        Bar(init_opts=opts.InitOpts(width="100%", height="900px")).add_xaxis(xs)
            .add_yaxis('spearman', ys1, category_gap=0)
            .add_yaxis('pearson', ys2, category_gap=0)
            .add_yaxis('kendall', ys3, category_gap=0)
            .add_yaxis('euclidean', ys4, category_gap=0)
            .add_yaxis('manhattan', ys5, category_gap=0)
            .add_yaxis('chebyshev', ys6, category_gap=0)
            .set_global_opts(
            # legend_opts=opts.LegendOpts(is_show=False),
            title_opts=opts.TitleOpts(title="histogram_" + cancer)
        ).render(ChartPathGetter().get_path(cancer, method, 'histogram.html'))
    )


if __name__ == '__main__':
    for c in ['luad','brca']:
        draw_histogram(c,'hsic+dge')
