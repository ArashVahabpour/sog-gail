### UTILS.py >>> VAE GAIL better vis

# latent_codes = torch.randn(5, args.latent_dim, device=device)
latent_codes = torch.load('/mnt/SSD3/arash/sog-gail/trained_models/circles/circles.vae-gail/Circles-v0_1140.pt')[2].cpu().numpy()
from sklearn.cluster import KMeans

latent_codes = KMeans(n_clusters=3).fit(latent_codes).cluster_centers_
latent_codes = torch.tensor(latent_codes, dtype=torch.float32, device=args.device)

#>>>>> plot
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

X = vae_modes
k = 3  # num clusters
X = np.vstack([KMeans(n_clusters=k).fit(X).cluster_centers_, X])
X_embedded = TSNE(n_components=2).fit_transform(X)
print(X_embedded.shape)

plt.scatter(X_embedded[k:,0], X_embedded[k:,1], c='b')  # points
plt.scatter(X_embedded[:k,0], X_embedded[:k,1], c='r')  # cluster centers
plt.show()




h = h5.File('/mnt/SSD3/arash/sog-gail/results/circles/sog-pretrain-10x-stronger/rewards.h5', 'r')
for k in h:
    if k!= 'expert':
        print(f"{k}\t{np.array(h[k]['mean']).mean()}\t{np.array(h[k]['std']).mean()}")

### SOG GAIL --> Circles
# 0       730.49      34.31
# 100     727.08      28.54
# 1000    769.28      99.46
# 1100    668.72      104.5
# 1200    651.79      32.15
# 1300    801.82      23.49
# 1400    605.34      77.24
# 1500    560.22      79.12
# 1600    633.72      70.20
# 1700    641.78      109.0
# 1800    771.33      50.60
# 1900    636.82      25.87
# 200     781.30      26.47
# 2000    705.94      11.69
# 2100    728.83      74.32
# 2200    768.01      60.09
# 2300    876.64      20.50
# 2400    847.93      30.28
# 2500    857.20      25.18
# 2600    823.43      24.57
# 2700    782.49      27.13
# 2800    793.29      21.00
# 2900    870.77      36.21
# 300     614.07      20.67
# 3000    877.55      21.74
# 3100    905.90      26.80
# 3200    852.47      39.02
# 3300    815.65      61.95
# 3400    780.84      65.19
# 3500    752.79      71.21
# 3600    766.26      73.72
# 3700    793.63      58.46
# 3800    793.61      69.29
# 3900    759.95      72.58
# 400     634.13      84.58
# 4000    804.65      79.98
# 4100    769.79      73.01
# 4200    711.72      67.56
# 4300    752.16      86.22
# 4400    740.99      90.67
# 4500    719.12      96.30
# 4600    761.67      110.0
# 4700    810.29      69.35
# 4800    814.31      103.6
# 500     621.78      29.57
# 600     588.92      29.00
# 700     625.71      92.51
# 800     769.53      34.64
# 900     699.06      34.53

# InfoGAIL --> Circles
# 0       484.356606117769        94.8791728184936
# 100     89.57085799142544       1.54626907255231
# 1000    571.2903824615469       97.7360742345184
# 1100    572.2053769801          71.67059882611575
# 1200    606.3135192128614       87.0528659612045
# 1300    563.1214036991576       108.695983949658
# 1400    585.4645920126287       137.192925275134
# 1500    558.3249615693045       157.337034668177
# 1600    608.6351019812195       83.6757997565897
# 1700    570.3023590462861       125.479467082025
# 1800    497.09331720748446      83.5957329051993
# 1900    491.78063968809033      93.4258444237006
# 200     505.690788622938        48.3271011146507
# 2000    456.25114074454524      81.7764495663486
# 2100    496.6051951955435       106.595800380853
# 2200    388.27820925056784      119.465838249937
# 2300    451.4666917450515       108.010859332440
# 2400    422.8802631086564       86.2117488039827
# 2500    582.4022722567668       119.578152324616
# 2600    582.3996108801948       110.261471379461
# 2700    717.1111740836786       78.2133566937534
# 2800    635.9132059690979       164.7525557958
# 2900    666.9632345103042       171.070720076937
# 300     534.9685855104572       157.407120221896
# 3000    647.325720479614        193.186018936473
# 3100    474.8574047485433       77.7525526051934
# 3200    566.161811000724        156.479033118848
# 3300    658.8420152574514       140.964213970570
# 3400    635.5000682181899       41.9430250485095
# 3500    661.1311368905709       81.3092116070161
# 3600    766.03599087151         67.90532537333034
# 3700    744.2755364801438       103.103203605026
# 3800    693.6671212610896       101.593002411407
# 3900    643.1720914866177       103.781140312232
# 400     614.4239928068477       118.155055877626
# 4000    699.1916418406399       108.802784217233
# 4100    752.7094094724613       71.2353756886351
# 4200    710.8015026974817       82.3114736548004
# 4300    738.6321503621926       71.4939337598150
# 4400    752.5250563003302       77.5775677190581
# 4500    744.9562297011893       38.5648421794909
# 4600    731.0898375008146       67.3941139532541
# 4700    719.2064612027901       96.6532640586863
# 4800    701.0133436604123       35.9306756241193
# 500     558.8122014681427       50.0550695632550
# 600     564.7621631729941       29.3828062118738
# 700     644.6455979348281       67.0054226829142
# 800     726.6568578656126       54.2472040320299
# 900     604.6768818772604       73.1507179710772




for k in h.keys():
    print(f"{k}\t{np.array(h[k]['mutual_info'])}")
# 1000    0.9153
# 780     0.9200


max = -np.inf
for k in h.keys():
    if np.array(h[k]['mutual_info'])[0]>max:
        max = np.array(h[k]['mutual_info'])[0]
        print(f"{k}\t{np.array(h[k]['mutual_info'])}")

['/mnt/SSD3/arash/sog-gail/results/circles/infogail-pretrain/rewards.h5',
 '/mnt/SSD3/arash/sog-gail/results/circles/sog-pretrain-10x-stronger/rewards.h5',
 '/mnt/SSD3/arash/sog-gail/results/antdir/ad.vae-gail/rewards.h5',
 '/mnt/SSD3/arash/sog-gail/results/antdir/d.ad.i/rewards.h5',
 '/mnt/SSD3/arash/sog-gail/results/antdir/d.ad6.i.10x/rewards.h5',
 '/mnt/SSD3/arash/sog-gail/results/antdir/d.ad6.s.10x/rewards.h5',
 '/mnt/SSD3/arash/sog-gail/results/antdir/d.ad.i.0.1x/rewards.h5',
 '/mnt/SSD3/arash/sog-gail/results/antdir/d.ad.i.10x/rewards.h5',
 '/mnt/SSD3/arash/sog-gail/results/antdir/d.ad6.i.0.1x/rewards.h5',
 '/mnt/SSD3/arash/sog-gail/results/antdir/d.ad.s/rewards.h5',
 '/mnt/SSD3/arash/sog-gail/results/antdir/d.ad6.s.0.1x/rewards.h5',
 '/mnt/SSD3/arash/sog-gail/results/antdir/ad6.vae-gail/rewards.h5',
 '/mnt/SSD3/arash/sog-gail/results/antdir/d.ad.s.10x/rewards.h5',
 '/mnt/SSD3/arash/sog-gail/results/antdir/d.ad6.i/rewards.h5',
 '/mnt/SSD3/arash/sog-gail/results/antdir/d.ad.s.0.1x/rewards.h5',
 '/mnt/SSD3/arash/sog-gail/results/halfcheetahvel/c.hcv.s.not_sh.10x/rewards.h5']

for f in [...]:
    print(f.split('/')[-2])

    import h5py as h5
    h = h5.File(f, 'r')

    best_mu = best_std = -np.inf
    for k, v in h.items():
        mu, std = [np.array(v[kk]).mean() for kk in ('mean', 'std')]
        if mu > max:
            best_mu, best_std = mu, std
    print(f"{k}\t{best_mu} +/- {best_std}")

# ad.vae-gail
# 990     108.72800149432987 +/- 294.2748151557247
# d.ad.i
# 995     319.57295281960444 +/- 722.8508381202604
# d.ad6.i.10x
# 995     203.74772446053316 +/- 254.97398817303232
# d.ad6.s.10x
# 995     344.94202269798933 +/- 217.52342928661736
# d.ad.i.0.1x
# 995     196.3934675231543 +/- 364.88274315312884
# d.ad.i.10x
# 995     220.608828562712 +/- 296.3150029451258
# d.ad6.i.0.1x
# 995     202.9579410659179 +/- 207.7122513908051
# d.ad.s
# 995     239.8772206498571 +/- 157.21111641013812
# d.ad6.s.0.1x
# 995     325.3418183129707 +/- 81.00092798340067
# ad6.vae-gail
# 990     111.71139188191133 +/- 96.90360350215128
# d.ad.s.10x
# 995     856.4647653961093 +/- 300.47629493070895
# d.ad6.i
# 995     197.4028066232638 +/- 283.26121638878476
# d.ad.s.0.1x
# 995     1062.7947163400227 +/- 64.4574991061217

from sklearn.cluster import KMeans
from shutil import copy

f = '/mnt/SSD3/arash/sog-gail/trained_models/halfcheetahdir/hcd.vae-gail/HalfCheetahDir-v0_vae_modes.pt'
copy(f, f+'.bak')
vae_modes = torch.load(f, map_location='cpu')
c = KMeans(n_clusters=2).fit(vae_modes.numpy()).cluster_centers_
c = torch.tensor(c, dtype=torch.float32)

torch.save((f, None, c), f)


#========
import h5py as h5
import glob
for f in list(glob.glob('*/rewards.h5')):
    m1,m2 = np.inf * (-1), None
    print(f)
    h = h5.File(f, 'r')
    for k,v in h.items():
        x, y = [np.array(v[s]).mean() for s in ('mean', 'std')]
        if x > m1:
            m1,m2 = x,y
            print(f"{f.split('/')[0]} / {k} / {x} +- {y}")
# =====
import h5py as h5

exp_name, epoch = 'hcv.vae-gail.20', '90'
h = h5.File(f'/mnt/SSD3/arash/sog-gail/results/halfcheetahvel/{exp_name}/rewards.h5', 'r')

plt.plot(np.unique(np.array(h[epoch]['all_x']).ravel()), np.array(h[epoch]['vel_mean']).ravel())
plt.fill_between(np.unique(np.array(h[epoch]['all_x']).ravel()), np.array(h[epoch]['vel_mean']).ravel() - np.array(h[epoch]['vel_std']).ravel(), np.array(h[epoch]['vel_mean']).ravel() + np.array(h[epoch]['vel_std']).ravel(), alpha=0.2)
plt.show()

h.close()

#=======
d = torch.load('./trajs_circles.pt')
s = d['states']
s2 = torch.cat([torch.zeros(500, 10, 10), s], dim=1)
s3 = torch.zeros(500,1000,30)
for i in range(500):
    print(i)
    for j in range(1000):
        s3[i][j] = torch.cat([s2[i][j+5*x] for x in range(3)])
d['state'] = s3
torch.save(d, './trajs_circles_longer.pt')
