import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return torch.sigmoid(x)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        
        self.attentions1 = [SpGraphAttentionLayer(nhid * nheads, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
            
        for i, attention in enumerate(self.attentions1):
            self.add_module('attention1_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)
        

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(torch.cat([att(x, adj) for att in self.attentions], dim=1))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x

class SpGATv2(nn.Module):
    """
    Sparse GATv2 network constructed from SpGraphAttentionLayerV2.

    Preserves the existing architecture footprint (multi-head, dropout,
    out-attention stage) to remain compatible with current code paths.
    """
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(SpGATv2, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayerV2(nfeat,
                                                   nhid,
                                                   dropout=dropout,
                                                   alpha=alpha,
                                                   concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_v2_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayerV2(nhid * nheads,
                                               nclass,
                                               dropout=dropout,
                                               alpha=alpha,
                                               concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(torch.cat([att(x, adj) for att in self.attentions], dim=1))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x

class GraphAttentionLayer(nn.Module):
    """
    稠密图注意力层（GAT）。
    实现方式：对每对节点计算注意力权重，再与邻居节点特征加权求和。
    参考论文：https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout          # 注意力权重的 dropout 率
        self.in_features = in_features  # 输入特征维度
        self.out_features = out_features  # 输出特征维度
        self.alpha = alpha              # LeakyReLU 的负斜率
        self.concat = concat            # 是否使用 ELU 激活（最后一层通常设为 False）

        # 可学习参数：节点特征线性变换矩阵 W
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # 可学习参数：注意力机制中的拼接权重向量 a
        # a^T [W h_i || W h_j]  -> 标量注意力分数
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """
        前向传播。
        参数：
            h:   (N, in_features)  节点特征矩阵，N 为节点数
            adj: (N, N)              邻接矩阵（稠密），0/1 或 0/权重
        返回：
            out: (N, out_features)  经注意力加权后的节点特征
        """
        # 1. 线性变换：h -> Wh
        Wh = torch.mm(h, self.W)          # (N, in_features) @ (in_features, out_features) -> (N, out_features)

        # 2. 计算未归一化的注意力分数 e
        e = self._prepare_attentional_mechanism_input(Wh)  # (N, N)

        # 3. 掩码：非邻居位置置为极小值，后续 softmax 时权重≈0
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)     # (N, N)

        # 4. 按行 softmax 得到归一化注意力权重
        attention = F.softmax(attention, dim=1)           # (N, N)

        # 5. dropout 作用于注意力权重（模型训练时生效）
        attention = F.dropout(attention, self.dropout, training=self.training)

        # 6. 加权求和：注意力矩阵 @ 变换后的节点特征
        h_prime = torch.matmul(attention, Wh)             # (N, N) @ (N, out_features) -> (N, out_features)

        # 7. 根据 concat 标志决定是否使用 ELU 激活
        if self.concat:
            return F.elu(h_prime)      # (N, out_features)
        else:
            return h_prime             # (N, out_features)

    def _prepare_attentional_mechanism_input(self, Wh):
        """
        计算未归一化的注意力分数 e_ij。
        采用拼接方式：e_ij = LeakyReLU( a^T [W h_i || W h_j] )
        参数：
            Wh: (N, out_features)
        返回：
            e:  (N, N)  注意力分数矩阵
        """
        # Wh1: 所有节点作为“源节点”时的投影  (N, 1)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        # Wh2: 所有节点作为“目标节点”时的投影  (N, 1)
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])

        # 广播相加：Wh1 + Wh2^T 得到任意两两节点间的分数  (N, N)
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)         # (N, N)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
    
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        if adj.layout == torch.sparse_coo:
            edge = adj.indices()
        else:
            edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class SpGraphAttentionLayerV2(nn.Module):
    """
    Sparse GATv2 layer (ICLR 2022, arXiv:2105.14491).

    Key GATv2 modifications vs. original GAT:
    - Dynamic attention (strict concatenation): e_ij = a^T LeakyReLU(W_pair · [h_i || h_j])
      (replaces static a^T [W h_i || W h_j] concatenation and avoids static attention).
    - Single shared linear map W for node value transform, and an attention vector a of size out_features.
    - Same sparse computation path and normalization (row-wise softmax-equivalent)
      using SpecialSpmm for efficiency on large sparse graphs.
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayerV2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # Node value transform (shared W)
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        # Strict concatenation attention transform: W_pair maps [Wh_i || Wh_j] -> out_features
        self.W_pair = nn.Parameter(torch.zeros(size=(2 * out_features, out_features)))
        nn.init.xavier_normal_(self.W_pair.data, gain=1.414)

        # Attention vector over transformed pair
        self.a = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        if adj.layout == torch.sparse_coo:
            edge = adj.indices()
        else:
            edge = adj.nonzero().t()

        # Linear transform for node values
        h = torch.mm(input, self.W)
        assert not torch.isnan(h).any()

        # Strict concatenation before attention projection
        # edge_cat shape: E x (2*out_features)
        edge_cat = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1)

        # Score: a^T LeakyReLU( W_pair · [Wh_i || Wh_j] )
        edge_e_raw = self.leakyrelu(edge_cat.mm(self.W_pair)).mm(self.a).squeeze()
        assert not torch.isnan(edge_e_raw).any()

        # Positive weights then row-normalization (softmax-equivalent)
        edge_e = torch.exp(edge_e_raw)
        assert not torch.isnan(edge_e).any()

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        assert not torch.isnan(h_prime).any()

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class SpecialSpmmFunction(torch.autograd.Function):
    """
    自定义稀疏矩阵乘法（Sparse-dense matmul）的自动求导函数。
    仅对稀疏矩阵的非零元素区域计算梯度，显著节省显存与计算量。
    用法：
        1. 前向：稀疏矩阵 A（COO 格式）与稠密矩阵 B 相乘，返回 A @ B。
        2. 反向：
           - 若需要对稀疏矩阵的 values 求梯度，只提取与 A 非零索引对应的梯度。
           - 若需要对 B 求梯度，直接返回 A^T @ grad_output。
    典型场景：GAT 系列模型中，注意力权重与节点特征做稀疏矩阵乘法时，
    通过该函数避免实例化完整的 N×N 注意力矩阵，从而支持大规模图训练。
    """
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
