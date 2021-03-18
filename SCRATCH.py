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
X = np.vstack([KMeans(n_clusters=3).fit(X).cluster_centers_, X])
X_embedded = TSNE(n_components=2).fit_transform(X)
print(X_embedded.shape)

plt.scatter(X_embedded[3:,0], X_embedded[3:,1], c='b')  # points
plt.scatter(X_embedded[:3,0], X_embedded[:3,1], c='r')  # cluster centers
plt.show()
