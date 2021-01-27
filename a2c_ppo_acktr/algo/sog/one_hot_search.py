import torch
from .base_search import BaseSearch


class OneHotSearch(BaseSearch):
    """
    Search over possible one-hot latent codes.
    """

    def _one_hot_codes(self, batch_size):
        """
        Create possible one-hot codes for each batch sample

        Returns:
            z: batch_size x n_latent x n_latent
        """

        z = torch.eye(self.latent_dim, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
        return z

    def resolve_latent_code(self, state, action):
        """
        Find the loss between the optimal fake data and the real data.

        Args:
            state: batch_size x dim_1 x ... x dim_kx
            action: batch_size x dim_1 x ... x dim_ky

        Returns:
            best_z: batch_size x n_latent
        """

        batch_size = action.shape[0]  # to accommodate for the end of the dataset when batch size might change
        # batch_size x latent_batch_size x n_latent
        all_z = self._one_hot_codes(batch_size)

        best_z = self.search_iter(all_z, state, action)

        return best_z
