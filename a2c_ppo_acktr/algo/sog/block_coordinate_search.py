import torch
import itertools
from .base_search import BaseSearch


class BlockCoordinateSearch(BaseSearch):
    """
    Block coordinate grid search optimizer over the distribution of points
    in the latent space.
    """

    def __init__(self, actor_critic, args):
        super().__init__(actor_critic, args)
        self.block_size = args.block_size
        self.n_rounds = args.n_rounds

    def _sample(self, old_z, block_idx):
        """
        Takes the best codes and perturbs
        Take old optimum code and repeat code 'latent_batch_size' times
        Then sample 'block_size' blocks from a normal distribution

        Args:
            old_z: batch_size x n_latent

        Returns:
            new_z: batch_size x latent_batch_size x n_latent
        """
        new_z = old_z.unsqueeze(1).repeat(1, self.latent_batch_size, 1)
        new_z[:, :, block_idx * self.block_size:(block_idx + 1) * self.block_size].normal_()

        return new_z

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
        best_z = torch.zeros(batch_size, self.latent_dim, device=self.device)
        # Go back over the latent vector and re-search 
        for round_idx, block_idx in itertools.product(range(self.n_rounds),
                                                      range(self.latent_dim // self.block_size)):
            # batch_size x latent_batch_size x n_latent
            new_z = self._sample(best_z, block_idx)
            best_z = self.search_iter(new_z, actions=action, states=state)

        return best_z
