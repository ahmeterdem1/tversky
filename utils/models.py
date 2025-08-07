import torch
import torch.nn as nn

# Tversky Aggregation Operations
tversky_dot = lambda a, b, dim: torch.sum(a * b, dim=dim)
tversky_sum = lambda a, b, dim: torch.sum(a + b, dim=dim)
tversky_max = lambda a, b, dim: torch.sum(torch.maximum(a, b), dim=dim)
tversky_min = lambda a, b, dim: torch.sum(torch.minimum(a, b), dim=dim)
tversky_mean = lambda a, b, dim: torch.sum(torch.mean(torch.stack((a, b)), dim=1), dim=dim)

class TverskyIntersection(nn.Module):

    def __init__(self,
                 features: torch.Tensor,
                 method: str = "product"):
        super().__init__()
        self.method = method.lower()

        if self.method == "product":
            self.op = tversky_dot
        elif self.method == "sum":
            self.op = tversky_sum
        elif self.method == "max":
            self.op = tversky_max
        elif self.method == "min":
            self.op = tversky_min
        elif self.method == "mean":
            self.op = tversky_mean
        else:
            raise ValueError("Invalid method. Choose from 'product', 'sum', 'max', 'min', or 'mean'.")

        self.features = features

    def forward(self, a: torch.Tensor, b: torch.Tensor):

        # a=(1, B, D) b=(P, 1, D) features = (1, K, D)

        common = torch.broadcast_shapes(a.size(), b.size())  # (P, B, D)
        b = b.expand(common)

        f_a = torch.matmul(a, self.features.mT)  # (1, B, K)
        f_b = torch.matmul(b, self.features.mT)  # (P, B, K)

        mask = ((f_a > 0) & (f_b > 0)).type(f_a.dtype)  # (P, B, K)
        f_a = f_a * mask
        f_b = f_b * mask

        return self.op(f_a, f_b, dim=-1).T  # (B, P)

class TverskyDifference(nn.Module):

    def __init__(self,
                 features: torch.Tensor,
                 method: str = "ignorematch"):
        super().__init__()
        self.method = method.lower()
        self.features = features

        # This is to prevent checking what "method" is at each forward propagation
        if self.method == "ignorematch":
            self.forward = self.forward_wrap_ignorematch
        elif self.method == "substractmatch":
            self.forward = self.forward_wrap_substractmatch
        else:
            raise ValueError(
                "Invalid method. Choose from 'ignorematch' or 'substractmatch'.")

    def forward_wrap_ignorematch(self, a, b):
        common = torch.broadcast_shapes(a.size(), b.size())  # (P, B, D)
        b = b.expand(common)

        f_a = a @ self.features.mT  # (B, K)
        f_b = b @ self.features.mT

        mask = ((f_a > 0) & (f_b <= 0)).type(f_a.dtype)
        f_a = f_a * mask
        return torch.sum(f_a, dim=-1).T

    def forward_wrap_substractmatch(self, a, b):
        common = torch.broadcast_shapes(a.size(), b.size())  # (P, B, D)
        b = b.expand(common)

        f_a = a @ self.features.mT  # (B, K)
        f_b = b @ self.features.mT

        mask = ((f_a > 0) & (f_b > 0) & (f_a > f_b)).type(f_a.dtype)
        f_a = f_a * mask
        f_b = f_b * mask

        return torch.sum(f_a - f_b, dim=-1).T

class TverskySimilarityLayer(nn.Module):

    def __init__(self,
                 in_dims: int,
                 out_dims: int,
                 num_features: int,
                 intersect_method: str = "product",
                 difference_method: str = "ignorematch"):
        super().__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.num_features = num_features
        self.intersect_method = intersect_method
        self.difference_method = difference_method

        self.features = nn.Parameter(torch.randn(size=(1, self.num_features, self.in_dims)), requires_grad=True)
        self.projection = nn.Parameter(torch.randn(size=(self.out_dims, 1, self.in_dims)), requires_grad=True)
        self.constants = nn.Parameter(torch.randn(size=(1, 1, 3)), requires_grad=True)

        self.intersection = TverskyIntersection(features=self.features, method=self.intersect_method)
        self.difference = TverskyDifference(features=self.features, method=self.difference_method)

    def forward(self, a: torch.Tensor):
        a = a.unsqueeze(dim=0)
        intersection_ = self.intersection(a, self.projection)
        diff1_ = self.difference(a, self.projection)
        diff2_ = self.difference(self.projection, a)

        t = torch.stack((intersection_, diff1_, diff2_), dim=1)  #(B, 3, out)
        out = self.constants @ t
        return out.squeeze()

class Backbone(nn.Module):

    def __init__(self):
        super().__init__()

        self.sequence = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1, bias=True),  #(32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=True),  #(64, 7, 7)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU()
        )

    def forward(self, x):
        return self.sequence(x)

class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = Backbone()
        self.tversky = TverskySimilarityLayer(in_dims=128,
                                              out_dims=10,
                                              num_features=64)

    def forward(self, x):
        x = self.backbone(x)
        return self.tversky(x)
