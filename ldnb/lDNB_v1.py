import numpy as np
import pandas as pd
import scipy.stats as stat

from tools.PathTools import LogPathGetter


def ssn_score(deta, pcc, nn):
    if pcc == 1:
        pcc = 0.99999999
    if pcc == -1:
        pcc = -0.99999999
    z = deta / ((1 - pcc * pcc) / (nn - 1))
    return z


def parallel_procedure(normal, disease, ref, sd_mean, j, refnum, pvalue):
    network = {}
    ssn = {}
    for p in ref.keys():
        t = p.split('_')
        r1 = ref[p]
        r2 = stat.pearsonr(normal[t[0]] + [disease[t[0]][j]], normal[t[1]] + [disease[t[1]][j]])[0]
        r = r2 - r1
        z = ssn_score(r, r1, refnum)
        p_value = 1 - stat.norm.cdf(abs(z))
        if p_value < pvalue:
            r = r if r > 0 else -r
            ssn[p] = r
            ssn[t[1] + "_" + t[0]] = r

            if t[0] not in network.keys():
                network[t[0]] = []
            network[t[0]].append(t[1])

            if t[1] not in network.keys():
                network[t[1]] = []
            network[t[1]].append(t[0])
    ci = {}
    for p in network.keys():
        if len(network[p]) < 3:
            continue

        sd = abs(disease[p][j] - sd_mean[p][1]) / sd_mean[p][0]
        pcc_in = 0
        pcc_out = 0
        count = 0
        for q in network[p]:
            sd += abs(disease[q][j] - sd_mean[q][1]) / sd_mean[q][0]
            pcc_in += ssn[p + "_" + q]
            for m in network[q]:
                if m != p:
                    pcc_out += ssn[q + "_" + m]
                    count += 1
        sd /= len(network[p]) + 1
        pcc_in /= len(network[p])
        if count == 0:
            continue
        pcc_out /= count
        if pcc_out == 0:
            continue
        ci[p] = [sd * pcc_in / pcc_out, sd, pcc_in, pcc_out]

    ci = sorted(ci.items(), key=lambda d: d[1][0], reverse=True)

    result = []
    for k in range(len(ci)):
        result.append([ci[k][0], ci[k][1][0], ci[k][1][1], ci[k][1][2], ci[k][1][3]])
    return pd.DataFrame(result, columns=['gene', 'Ic', 'sd', 'pcc_in', 'pcc_out'])


def reference_network(normal):
    keys = list(normal.keys())
    n = len(keys)
    result = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            r = stat.pearsonr(normal[keys[i]], normal[keys[j]])
            # The threshold of P-value need be set in here for Pearson Correlation Coefficient
            # if r[1] < 0.01 / (20):
            if r[1] < 0.01:
                result.append([keys[i], keys[j], r[0]])
    return pd.DataFrame(result, columns=['i', 'j', 'pearsonr'])


def tipping_point(sorted_list: pd.DataFrame, sorted_rdata: pd.DataFrame, title):
    pvalue = 0.05  # p-value threshold is set

    sorted_rdata = sorted_rdata.apply(np.exp2) - 1
    normal = {}
    normal_df = sorted_rdata.loc[:, sorted_list['flag'] == 0]
    for gene in normal_df.index:
        normal[gene] = normal_df.loc[gene, :].tolist()
    refnum = normal_df.index.size

    sd_mean = {}
    for key in normal.keys():
        sd_mean[key] = [np.std(normal[key]), np.mean(normal[key])]

    ref = {}
    reference_n = reference_network(normal)
    for _, row in reference_n.iterrows():
        t = row['i']
        ref[row['i'] + "_" + row['j']] = float(row['pearsonr'])

    disease = {}
    cancer_df = sorted_rdata.loc[:, sorted_list['flag'] != 0]
    for gene in cancer_df.index:
        disease[gene] = cancer_df.loc[gene, :].tolist()
    result = []
    for j in range(cancer_df.columns.size):
        resu = parallel_procedure(normal, disease, ref, sd_mean, j, refnum, pvalue)
        result.append(resu)
    ic_mean = list(map(lambda df: df['Ic'].mean(), result))
    ic_df = pd.Series(ic_mean, index=sorted_list[sorted_list['flag'] != 0]['stage'])
    if title is not None:
        top5_tip = pd.Series(ic_mean, index=range(len(ic_mean))).sort_values(ascending=False).head(5)
        lpg = LogPathGetter()
        barcodes = sorted_list[sorted_list['flag'] != 0].index.tolist()
        a = 0
        sets: list[set] = []
        for i in top5_tip.index:
            a += 1
            barcode = barcodes[i]
            df = result[i]
            sets.append(set(df.head(10)['gene'].tolist()))
            df.to_csv(lpg.get_path(title, a, barcode, '.csv'))
        set_r = sets[0].intersection(*sets[1:])
        print(title)
        print(' '.join(map(lambda a: a.split('|')[0], set_r)))
    return ic_df.fillna(0)
