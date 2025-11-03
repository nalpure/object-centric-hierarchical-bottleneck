import torch
from typing import Tuple, Union
from torch import nn, Tensor
from torch.nn import functional as F
from einops import reduce

class SlotAttention(nn.Module):
    def __init__(self, num_slots, slot_size, in_features, iters = 3, eps = 1e-8, mlp_hidden_size = 128):
        super().__init__()

        self.in_features = in_features
        self.iters = iters
        self.num_slots = num_slots
        self.slot_size = slot_size  # number of hidden layers in slot dimensions
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = eps

        self.norm_inputs = nn.LayerNorm(self.in_features)
        # I guess this is layer norm across each slot? should look into this
        self.norm_slots = nn.LayerNorm(self.slot_size)
        self.norm_mlp = nn.LayerNorm(self.slot_size)

        # Linear maps for the attention module.
        self.project_q = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_k = nn.Linear(self.in_features, self.slot_size, bias=False)
        self.project_v = nn.Linear(self.in_features, self.slot_size, bias=False)

        # OT approach
        self.mlp_weight_input = nn.Linear(self.in_features, 1)
        self.mlp_weight_slots = nn.Linear(self.slot_size, 1)

        # Slot update functions.
        self.gru = nn.GRUCell(self.slot_size, self.slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size),
        )

        self.slots_mu = nn.Parameter(nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")))
        self.slots_log_sigma = nn.Parameter(nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")))

        """self.register_buffer(
            "slots_mu",
            nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")),
        )
        self.register_buffer(
            "slots_log_sigma",
            nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")),
        )"""
    
    def step(self, slots, n_s, k, v, a, batch_size, num_inputs):
        slots_prev = slots
        slots = self.norm_slots(slots)
        
        b = self.mlp_weight_slots(slots).squeeze(-1).softmax(-1) * n_s  # <- this is new

        # Attention.
        q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
        assert_shape(q.size(), (batch_size, n_s, self.slot_size))

        # attn_norm_factor = self.slot_size ** -0.5
        # attn_logits = attn_norm_factor * torch.matmul(k, q.transpose(2, 1))
        # attn_logits = -attn_logits
        attn_logits = torch.cdist(k, q)  # <- this is new
        # k = k / k.norm(dim=2, keepdim=True)
        # q = q / q.norm(dim=2, keepdim=True)
        # attn_logits = 1.0 - 6* torch.matmul(k, q.transpose(2, 1))

        # attn = F.softmax(attn_logits, dim=-1)
        attn_logits, p, q = minimize_entropy_of_sinkhorn(attn_logits, a, b, mesh_lr=5)   # <- this is new
        attn, _, _ = sinkhorn(attn_logits, a, b, u=p, v=q)  # <- this is new

        # `attn` has shape: [batch_size, num_inputs, num_slots].
        assert_shape(attn.size(), (batch_size, num_inputs, n_s))

        # Weighted mean.
        # attn = attn + self.epsilon
        # attn = attn / torch.sum(attn, dim=1, keepdim=True)
        updates = torch.matmul(attn.transpose(1, 2), v)
        # `updates` has shape: [batch_size, num_slots, slot_size].
        assert_shape(updates.size(), (batch_size, n_s, self.slot_size))

        # Slot update.
        # GRU is expecting inputs of size (N,H) so flatten batch and slots dimension
        slots = self.gru(
            updates.view(batch_size * n_s, self.slot_size),
            slots_prev.view(batch_size * n_s, self.slot_size),
        )
        slots = slots.view(batch_size, n_s, self.slot_size)
        assert_shape(slots.size(), (batch_size, n_s, self.slot_size))
        slots = slots + self.mlp(self.norm_mlp(slots))
        assert_shape(slots.size(), (batch_size, n_s, self.slot_size))

        # return only input variables which changed
        return slots, attn

    def forward(self, inputs: torch.Tensor, slots_init=None):
        # `inputs` has shape [batch_size, num_inputs, inputs_size]. example: [256, 4096, 64]
        batch_size, num_inputs, inputs_size = inputs.shape
        n_s = self.num_slots
        n_i = self.iters

        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        assert_shape(k.size(), (batch_size, num_inputs, self.slot_size))
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        assert_shape(v.size(), (batch_size, num_inputs, self.slot_size))

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        if slots_init is not None:
            slots = slots_init.clone()
        else:
            slots_init = torch.randn((batch_size, n_s, self.slot_size))
            slots_init = slots_init.type_as(inputs)
            slots = self.slots_mu + self.slots_log_sigma.exp() * slots_init

        a = self.mlp_weight_input(inputs).squeeze(-1).softmax(-1) * n_s  # <- this is new

        # Multiple rounds of attention.
        for _ in range(n_i-1):
            slots, attn = self.step(slots, n_s, k, v, a, batch_size, num_inputs)
        
        slots = slots.detach() # detach for approx implicit diff
        
        # Final round of attention.
        slots, attn = self.step(slots, n_s, k, v, a, batch_size, num_inputs)            

        return slots, attn.permute(0,2,1)
    
@torch.jit.script
def sinkhorn(
    C: Tensor,
    a: Tensor,
    b: Tensor,
    n_sh_iters: int = 5,
    temperature: float = 1,
    u: Union[Tensor, None] = None,
    v: Union[Tensor, None] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    p = -C / temperature
    log_a = torch.log(a)
    log_b = torch.log(b)

    if u is None:
        u = torch.zeros_like(a)
    if v is None:
        v = torch.zeros_like(b)

    for _ in range(n_sh_iters):
        u = log_a - torch.logsumexp(p + v.unsqueeze(1), dim=2)
        v = log_b - torch.logsumexp(p + u.unsqueeze(2), dim=1)

    logT = p + u.unsqueeze(2) + v.unsqueeze(1)
    return logT.exp(), u, v


@torch.enable_grad()
def minimize_entropy_of_sinkhorn(
    C_0, a, b, noise=None, mesh_lr=1, n_mesh_iters=4, n_sh_iters=5, reuse_u_v=True
):
    if noise is None:
        noise = torch.randn_like(C_0)

    C_t = C_0 + 0.001 * noise
    C_t.requires_grad_(True)

    u = None
    v = None
    for i in range(n_mesh_iters):
        attn, u, v = sinkhorn(C_t, a, b, u=u, v=v, n_sh_iters=n_sh_iters)

        if not reuse_u_v:
            u = v = None

        entropy = reduce(
            torch.special.entr(attn.clamp(min=1e-20, max=1)), "n a b -> n", "mean"
        ).sum()
        (grad,) = torch.autograd.grad(entropy, C_t, retain_graph=True)
        grad = F.normalize(grad + 1e-20, dim=[1, 2])
        C_t = C_t - mesh_lr * grad

    # attn, u, v = sinkhorn(C_t, a, b, u=u, v=v, num_sink=num_sink_iters)

    if not reuse_u_v:
        u = v = None

    return C_t, u, v


def assert_shape(actual: Union[torch.Size, Tuple[int, ...]], expected: Tuple[int, ...], message: str = ""):
    assert actual == expected, f"Expected shape: {expected} but passed shape: {actual}. {message}"