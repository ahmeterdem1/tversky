import torch
import torch.nn as nn

# Tversky Aggregation Operations
tversky_dot = lambda a, b, dim: torch.sum(a * b, dim=dim)
tversky_sum = lambda a, b, dim: torch.sum(a + b, dim=dim)
tversky_max = lambda a, b, dim: torch.sum(torch.maximum(a, b), dim=dim)
tversky_min = lambda a, b, dim: torch.sum(torch.minimum(a, b), dim=dim)
tversky_mean = lambda a, b, dim: torch.sum(torch.mean(torch.stack((a, b)), dim=1), dim=dim)

class TverskyIntersection(nn.Module):

    """
        This module implements the Tversky intersection operation defined by the paper.
        Given vectors a and b, both are multiplied by a set of learnable vector features,
        to result in scalars. The scalars coming from a and b are aggregated if both
        scalars are non-negative.

        Aggregation methods implemented here are 'product', 'sum', 'max', 'min', and 'mean'.
        The paper defines others, but they are not implemented here.

        This class implements Equation 3 in the paper, "Feature Set Intersection".
    """

    def __init__(self,
                 features: torch.Tensor,
                 method: str = "product"):
        """
            Initialize Tversky Intersection Module.

            Args:
                features (Tensor): Learnable feature tensor of shape (1, K, D),
                    where K is the number of feature vectors, D is the vector
                    dimensionality. This tensor is supposed to be shared within
                    a full Tversky Similarity Layer. It will also be used by the
                    set difference operators.

                method (str): Aggregation method to use. Options are 'product',
                    'sum', 'max', 'min', and 'mean'. Default is 'product'.
        """
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

    """
        This module implements the Tversky difference operation defined by the paper.
        Tversky difference produces the same scalars as Tversky intersection by multiplying
        object vectors a and b by a set of same dimensional feature vectors.

        From that point, authors define two methods to compute the difference:

        1. Ignore Match: If a scalar from a is positive and the corresponding scalar from b is not,
            sum scalars generated from a. This option implements Equation 4 in the paper, "Feature Set
            Difference (Ignore Match)".

        2. Substract Match: If a scalar from a is positive and the corresponding scalar from b is positive,
            but the scalar from a is greater than the scalar from b, then sum the differences of the scalars
            from a and b. This option implements Equation 5 in the paper, "Feature Set Difference (Substract Match)".

    """

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

    """
        This module implements the Tversky projection defined by the paper. It hosts a
        set of feature vectors and a set of projection vectors. It hosts a Tversky
        intersection and a Tversky difference operator, which are used to compute the
        Tversky similarity given by Equation 1 in the paper.

        "a" vectors are inputs to the layer, "b" vectors are the set of projection vectors.
        Given a batch of "a" vectors, "b" vectors are broadcasted to match the shape during
        operations. This layer takes a batch of vectors with shape (B, in_dims) and outputs
        a batch of vectors of shape (B, out_dims), the same way as a linear layer would do.
        Therefore, any linear layer in any network could be replaced by this module.

        The outputs of intersection, A-B difference and B-A difference are combined with
        learned parameters of shape (1, 1, 3). These are given as theta, alpha and beta in
        the paper.

        Parameters are initialized with normal distribution by default. Uniform or orthogonal
        initialization can also be used by the corresponding parameter.

    """

    def __init__(self,
                 in_dims: int,
                 out_dims: int,
                 num_features: int,
                 intersect_method: str = "product",
                 difference_method: str = "ignorematch",
                 initialization: str = "normal"):
        super().__init__()

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.num_features = num_features
        self.intersect_method = intersect_method
        self.difference_method = difference_method
        self.initialization = initialization.lower()

        self.features_ = torch.empty(size=(1, self.num_features, self.in_dims))
        self.projection_ = torch.empty(size=(self.out_dims, 1, self.in_dims))
        self.constants_ = torch.empty(size=(1, 1, 3))

        if self.initialization == "normal":
            nn.init.normal_(self.features_, mean=0.0, std=1.0)
            nn.init.normal_(self.projection_, mean=0.0, std=1.0)
            nn.init.normal_(self.constants_, mean=0.0, std=1.0)

        elif self.initialization == "uniform":
            nn.init.uniform_(self.features_, a=-1.0, b=1.0)
            nn.init.uniform_(self.projection_, a=-1.0, b=1.0)
            nn.init.uniform_(self.constants_, a=-1.0, b=1.0)

        elif self.initialization == "orthogonal":
            nn.init.orthogonal_(self.features_, gain=1)
            nn.init.orthogonal_(self.projection_, gain=1)
            nn.init.orthogonal_(self.constants_, gain=1)

        else:
            raise ValueError("Invalid initialization method. Choose from 'normal', 'uniform', or 'orthogonal'.")

        self.features = nn.Parameter(self.features_, requires_grad=True)
        self.projection = nn.Parameter(self.projection_, requires_grad=True)
        self.constants = nn.Parameter(self.constants_, requires_grad=True)

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

    def __init__(self, out_dim: int = 128, in_channels: int = 1, in_width: int = 28):
        super().__init__()
        self.out_dim = out_dim
        self.in_channels = in_channels
        self.in_width = in_width

        self.sequence = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=4, stride=2, padding=1, bias=True),  #(32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=True),  #(64, 7, 7)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * self.in_width * self.in_width // 16, self.out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.sequence(x)

class TverskyModel(nn.Module):

    def __init__(self,
                 intermediate_dim: int = 128,
                 out_dim: int = 10,
                 num_tversky_features: int = 64,
                 in_channels: int = 1,
                 in_width: int = 28,
                 *args, **kwargs):
        super().__init__()

        self.intermediate_dim = intermediate_dim
        self.out_dim = out_dim
        self.num_tversky_features = num_tversky_features

        self.backbone = Backbone(out_dim=self.intermediate_dim,
                                 in_channels=in_channels,
                                 in_width=in_width)
        self.tversky = TverskySimilarityLayer(in_dims=self.intermediate_dim,
                                              out_dims=self.out_dim,
                                              num_features=self.num_tversky_features,
                                              *args, **kwargs)

    def forward(self, x):
        return self.tversky(self.backbone(x))

class BaseModel(nn.Module):

    def __init__(self,
                 intermediate_dim: int = 128,
                 out_dim: int = 10,
                 in_channels: int = 1,
                 in_width: int = 28):
        super().__init__()
        self.intermediate_dim = intermediate_dim
        self.out_dim = out_dim

        self.backbone = Backbone(out_dim=intermediate_dim,
                                 in_channels=in_channels,
                                 in_width=in_width)
        self.classifier = nn.Linear(in_features=intermediate_dim, out_features=out_dim)

    def forward(self, x):
        return self.classifier(self.backbone(x))
