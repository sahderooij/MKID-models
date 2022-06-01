import numpy as np
import pandas as pd
import scipy.io


def get_Sondata(filename):
    with open(filename, 'r') as file:
        Sonfile = []
        line = file.readline()
        while line != "":
            dataset = {'Header': [], 'Params': {}, 'Data': []}
            while (line.split(' ')[0] not in ['R', 'TERM', 'FTERM'] and '=' not in line) and line != "":
                if line.split(' ')[0] == 'FTERM':
                    for i in range(3):
                        dataset['Header'].append(line.replace('\n', ''))
                        line = file.readline()
                else:
                    dataset['Header'].append(line.replace('\n', ''))
                    line = file.readline()
            dataset['Header'].append(line.replace('\n', ''))
            colnames = file.readline().replace('\n', '').split(',')
            dataset['Data'] = pd.DataFrame(columns=colnames)
            line = file.readline()
            while "=" in line:
                Param, Val = line.replace(' ', '').split('=')
                dataset['Params'][Param] = float(Val)
                line = file.readline()
            ind = 0
            while np.fromstring(line, sep=',').size == len(colnames):
                dataset['Data'].loc[ind] = np.fromstring(
                    line.replace('\n', ''), sep=","
                )
                ind += 1
                line = file.readline()
            Sonfile.append(dataset)
    return Sonfile


def gen_Qdesign(Soncsv, savemat=False, matname="Qdesign"):
    # Make Qdesign array for Matlab select_Classicdesign()
    numcol = 1 + len(Soncsv)
    numrow = 1 + len(Soncsv[0]["Data"].index)
    Qdesign = np.zeros((numrow, numcol))
    Qdesign[0, 1:] = [Soncsv[i]["Params"]["Lc"] for i in range(len(Soncsv))]
    Qdesign[1:, 0] = Soncsv[0]["Data"]["Frequency (GHz)"].values
    for i, dataset in enumerate(Soncsv):
        Lc = dataset["Params"]["Lc"]
        for j, freq in enumerate(dataset["Data"]["Frequency (GHz)"].values):
            if freq in Qdesign[:, 0] and Lc in Qdesign[0, :]:
                Qdesign[Qdesign[:, 0] == freq, Qdesign[0, :] == Lc] = np.log10(
                    np.pi / (2 * (10 ** (dataset["Data"]["DB[S13]"].iloc[j] / 20)) ** 2)
                )

    assert (
        0 not in Qdesign[1:, 1:]
    ), "Data not complete: probably ABS frequency Sonnet data"
    if savemat:
        scipy.io.savemat(matname + ".mat", dict(Qdesign=Qdesign))
    return Qdesign
