import torch
import torch.nn as nn


class NT_Xent(nn.Module):
    def __init__(self, temperature=1.0):
        super(NT_Xent, self).__init__()
        self.temperature = temperature
        self.sim_f = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss()

    def _forward_single(self, z1, z2):
        """
        Compute the NT-Xent loss between two sets of embeddings with optional masking.

        Args:
            z1, z2: Tensors of shape (N, D) representing embeddings from two views.

        Returns:
            Loss value (scalar).
        """
        assert z1.size() == z2.size(), "z1 and z2 must have the same shape."
        N = z1.size(0)

        # Create validity masks: True for valid samples (i.e. not masked)
        valid1 = ~torch.isnan(z1[..., 0])
        valid2 = ~torch.isnan(z2[..., 0])

        # Only keep embeddings where both masks indicate validity.
        valid = valid1 & valid2
        z1 = z1[valid]
        z2 = z2[valid]
        N = z1.size(0)  # This is N' now
        if N == 0:
            return torch.tensor(0.0, device=z1.device)

        # Concatenate embeddings from both views.
        z = torch.cat((z1, z2), dim=0)  # shape: (2N', D)
        sim = self.sim_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature  # (2N', 2N')

        # Extract positive similarities: off-diagonals between z1 and z2.
        sim_i_j = torch.diag(sim, N)
        sim_j_i = torch.diag(sim, -N)

        # Create a mask to remove self-similarities.
        full_mask = torch.eye(2 * N, dtype=torch.bool, device=z1.device)
        for i in range(N):
            full_mask[i, i + N] = True
            full_mask[i + N, i] = True

        positives = torch.cat((sim_i_j, sim_j_i), dim=0).view(-1, 1)  # (2N', 1)
        negatives = sim[~full_mask].view(2 * N, -1)

        logits = torch.cat((positives, negatives), dim=1)  # (2N', 1 + (2N' - 1))
        labels = torch.zeros(2 * N, dtype=torch.long, device=z1.device)

        loss = self.criterion(logits, labels)
        return loss

    def forward(self, z1, z2):
        """
        z1, z2: tensors of shape (B, N, D) or (N, D).
        """
        if z1.dim() == 3:  # batched mode
            B = z1.size(0)
            losses = []
            for b in range(B):
                loss_b = self._forward_single(z1[b], z2[b])
                losses.append(loss_b)
            return torch.stack(losses).mean()
        else:
            return self._forward_single(z1, z2)


if __name__ == "__main__":
    lossfunc = NT_Xent(0.5)

    # Case 1: Perfect match.
    z1 = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float)
    z2 = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float)
    loss = lossfunc(z1, z2)
    print("Case 1: Perfect match. Loss={}".format(loss.item()))

    # Case 2: Perfect mismatch.
    z1 = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    z2 = torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=float)
    loss = lossfunc(z1, z2)
    print("Case 2: Perfect mismatch. Loss={}".format(loss.item()))

    # Case 3: Partial match.
    z1 = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    z2 = torch.tensor(
        [[0.9, 0.1, 0, 0], [0.1, 0.9, 0, 0], [0, 0, 0.9, 0.1], [0, 0, 0.1, 0.9]], dtype=float
    )
    loss = lossfunc(z1, z2)
    print("Case 3: Partial match. Loss={}".format(loss))

    # Case 4: With masks (manual example).
    z1 = torch.tensor([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=float)
    z2 = torch.tensor([[0.9, 0.1, 0, 0], [0, 0, 0.1, 0.9]], dtype=float)
    loss = lossfunc(z1, z2)
    print(
        "Case 4: Partial match with masking outside of loss function. Loss={}".format(loss.item())
    )

    # Case 5: With masks (auto).
    z1 = torch.tensor(
        [
            [1, 0, 0, 0],
            [float("nan"), float("nan"), float("nan"), float("nan")],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    z2 = torch.tensor(
        [
            [0.9, 0.1, 0, 0],
            [0.1, 0.9, 0, 0],
            [float("nan"), float("nan"), float("nan"), float("nan")],
            [0, 0, 0.1, 0.9],
        ],
        dtype=float,
    )
    mask1 = torch.tensor([0, 1, 0, 0], dtype=torch.int)
    mask2 = torch.tensor([0, 0, 1, 0], dtype=torch.int)
    loss = lossfunc(z1, z2)
    print("Case 5: Partial match with masking in loss function. Loss={}".format(loss.item()))
