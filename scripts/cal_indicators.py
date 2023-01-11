import os   
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

def collect_depth(d_error, d_acc, tdepth, pdepth):
    de = tdepth-pdepth
    for i in list(range(0, 15, 1)):
        if i*10<tdepth<=(i+1)*10: # 0~10m
            d_error.append(de)
            d_acc[i].append(1-abs(de)/tdepth)
            break
    return 

def norm_func(x, mu, sigma):
    pdf = np.exp(-((x - mu)**2) / (2* sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf

def plot_distribution(data:list, savePath:str):
    mean = np.mean(data)
    std = np.std(data)
    eage = ceil(max(abs(max(data)), abs(min(data))))

    plt.title('Distribution')
    plt.xlabel('Data Range(Total:{})'.format(str(len(data))))
    plt.ylabel('Probability')

    numb, _, _ = plt.hist(
        np.squeeze(np.asarray(data)),
        bins=eage*2,
        density=1,
        facecolor='green',
        rwidth=0.9,
        alpha=0.6
    )

    x = np.arange(-eage, eage, 0.1)
    y = norm_func(x, mean, std)
    plt.plot(x, y, '--', color='black')

    one_sigma = 0 
    two_sigma = 0
    for e in data:
        if -std*2 <= e <= std*2:
            two_sigma += 1
            if -std <= e <= std:
                one_sigma += 1
    one_sigma = 100*one_sigma/len(data)
    two_sigma = 100*two_sigma/len(data)

    plt.text(-eage+1, max(numb), 
            "mu={}, sigma={}".format(str(round(mean,3)), str(round(std,3))), 
            fontsize=8)
    plt.text(mean+1, max(numb), 
            "1_sigma={}%, 2_sigma={}%".format(str(round(one_sigma,3)), str(round(two_sigma,3))), 
            fontsize=8)

    if not os.path.exists(savePath):
        os.mkdir(savePath)
    plt.savefig(os.path.join(savePath, 'Distribution.jpg'))

    return

def cal_depth_indicators(errors, accs):
    plot_distribution(errors, 'DepthIndicators')
    de_acc = [np.mean(a) for a in accs] # get accuracy error in different range
    len_stats = [len(a) for a in accs]
    depth_stats = [str(len_stats)+'\n', str(de_acc)+'\n']
    with open('DepthIndicators/depth-indicators.txt', 'w', encoding='utf-8') as f:
        f.writelines(depth_stats)
    return
    