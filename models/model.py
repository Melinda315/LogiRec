import numpy as np
import torch as th
import torch
import torch.nn as nn
from scipy.sparse import load_npz
import scipy.sparse as sp
from utils.helper import default_device
import math
import torch.nn.functional as F
from geoopt import Lorentz
import geoopt.manifolds.lorentz.math as lorentz_math
from geoopt import PoincareBall
from geoopt import ManifoldParameter
from utils.helper import sparse_mx_to_torch_sparse_tensor, normalize
from torch.utils.data import DataLoader

import models.encoders as encoders

import geoopt as gt
import itertools
import geoopt.manifolds.stereographic.math as pmath
from scipy.sparse import *
from utils.train_utils import build_tree, generate_user_tag_matrix, find_exclusion
import time
from config import parser

eps = 1e-15
MIN_NORM = 1e-15
dropout = 0.5



args = parser.parse_args()
cuda_device = torch.device('cuda:0')
path = './data/'
dataset = args.dataset
implication = torch.load(path + dataset + '/implication.pt', map_location='cuda:0')
exclusion = torch.load(path + dataset + '/exclusion.pt', map_location='cuda:0')

class MobiusLinear(nn.Linear):  # hyperbolic linear layer 4.1
    def __init__(self, *args, hyperbolic_input=True, hyperbolic_bias=True, nonlin=None, c=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ball_ = gt.PoincareBall(c=c)
        if self.bias is not None:
            if hyperbolic_bias:
                self.bias = gt.ManifoldParameter(self.bias, manifold=self.ball_)
                with torch.no_grad():
                    self.bias.set_(pmath.expmap0(self.bias.normal_() * 1e-3, k=self.ball_.k))
        with torch.no_grad():
            fin, fout = self.weight.size()
            k = (6 / (fin + fout)) ** 0.5  # xavier uniform
            self.weight.uniform_(-k, k)
        self.hyperbolic_bias = hyperbolic_bias
        self.hyperbolic_input = hyperbolic_input
        self.nonlin = nonlin

    def forward(self, input):
        return mobius_linear(
            input,
            weight=self.weight,
            bias=self.bias,
            hyperbolic_input=self.hyperbolic_input,
            nonlin=self.nonlin,
            hyperbolic_bias=self.hyperbolic_bias,
            k=self.ball_.k,
        )

    def extra_repr(self):
        info = super().extra_repr()
        info += ", hyperbolic_input={}".format(self.hyperbolic_input)
        if self.bias is not None:
            info += ", hyperbolic_bias={}".format(self.hyperbolic_bias)
        return info

def mobius_linear(
    input,
    weight,
    bias=None,
    hyperbolic_input=True,
    hyperbolic_bias=True,
    nonlin=None,
    k=-1.0,
):
    if hyperbolic_input:
        weight = F.dropout(weight, dropout)
        output = pmath.mobius_matvec(weight, input, k=k)
    else:
        output = torch.nn.functional.linear(input, weight)
        output = pmath.expmap0(output, k=k)
    if bias is not None:
        if not hyperbolic_bias:
            bias = pmath.expmap0(bias, k=k)
        output = pmath.mobius_add(output, bias, k=k)
    if nonlin is not None:
        output = pmath.mobius_fn_apply(nonlin, output, k=k)
    output = pmath.project(output, k=k)
    return output

class LogiRec(nn.Module):

    def __init__(self, users_items, args, feature_num, hidden_size, embed_dim, num_tag, **kwargs):
        super(LogiRec, self).__init__(**kwargs)

        self.c = torch.tensor([args.c]).to(default_device())

        self.manifold = Lorentz(args.c)
        self.ball = PoincareBall(args.c)
        self.encoder = getattr(encoders, "HG")(args.c, args)

        self.num_users, self.num_items = users_items
        self.margin = args.margin
        self.weight_decay = args.weight_decay
        self.args = args


        self.ball_ = gt.PoincareBall(c=1.0)
        points = torch.randn(num_tag, embed_dim) * 1e-5
        points = pmath.expmap0(points.to(cuda_device), k=self.ball_.k)
        self.emb_tag = gt.ManifoldParameter(points, manifold=self.ball_)

        self.encoder_HMI = nn.Sequential(
            MobiusLinear(feature_num, embed_dim, bias=True, nonlin=None),
        )

        self.vtg = nn.Embedding(num_embeddings=self.num_items,
                                embedding_dim=args.embedding_dim).to(default_device())
        self.vtg.state_dict()['weight'].normal_(mean=0, std=args.scale)
        self.vtg.weight = nn.Parameter(self.manifold.expmap0(self.vtg.state_dict()['weight']))
        self.vtg.weight = ManifoldParameter(self.vtg.weight, self.manifold, requires_grad=True)


        self.utg = nn.Embedding(num_embeddings=self.num_users,
                                embedding_dim=args.embedding_dim).to(default_device())
        self.utg.state_dict()['weight'].normal_(mean=0, std=args.scale)
        self.utg.weight = nn.Parameter(self.manifold.expmap0(self.utg.state_dict()['weight']))
        self.utg.weight = ManifoldParameter(self.utg.weight, self.manifold, requires_grad=True)

        self.lam = args.lam
        self.digit = args.digit
        self.eta, self.tag_levels = build_tree(implication)
        self.user_con = None
        self.item_tag_matrix = None
        self.user_item_matrix = None
        self.item_ids = None


    def encode(self, adj):
        adj = adj.to(default_device())

        emb_utg = self.manifold.projx(self.utg.weight)
        emb_vtg = self.manifold.projx(self.vtg.weight)

        x2 = torch.cat([emb_utg, emb_vtg], dim=0)
        h_tg = self.encoder.encode(x2, adj)

        return h_tg

    def decode(self, h_all, idx):
        h = h_all
        emb_utg = h[idx[:, 0].long()]
        emb_vtg = h[idx[:, 1].long()]

        assert not torch.isnan(emb_utg).any()
        assert not torch.isinf(emb_vtg).any()
        sqdist = self.manifold.dist2(emb_utg, emb_vtg, keepdim=True).clamp_max(15.0)

        return sqdist


    def calc_alpha(self, emb_u, user_ids):

        gr = self.calc_gr(emb_u)
        con = self.calc_con(user_ids)
        alpha = torch.sqrt(con * gr)

        return alpha

    def calc_con(self, user_ids):
        user_ids = user_ids.cpu()
        if self.user_con == None:
            user_tag_matrix = generate_user_tag_matrix(self.user_item_matrix, self.item_tag_matrix)
            user_tag_matrix = torch.from_numpy(user_tag_matrix).float().cpu()
            user_exclusion_tag = find_exclusion(user_tag_matrix, exclusion)

            tfd = self.item_tag_matrix[self.item_ids].sum()
            tf = self.item_tag_matrix[self.item_ids].sum(dim=0)
            tag_TF = torch.log(tf + 1) / torch.log(tfd)
            tag_TF = tag_TF.numpy()

            user_con = np.zeros(user_tag_matrix.shape[0])
            for user, user_exclusion in enumerate(user_exclusion_tag):
                if len(user_exclusion) == 0:
                    user_con[user] = 0.
                    continue
                con = 0.
                for exclusion_tag in user_exclusion:
                    left = exclusion_tag[0]
                    llevel = self.tag_levels[left]
                    right = exclusion_tag[1]
                    rlevel = self.tag_levels[right]
                    if llevel != rlevel: continue
                    eta = self.eta
                    
                    con += tag_TF[left] * tag_TF[right] * math.pow(self.digit, (eta - llevel))

                user_con[user] = con
            self.user_con = torch.tensor(user_con).unsqueeze(1)
            self.user_con = torch.pow(self.digit, -self.user_con)

        user_cons = self.user_con[user_ids]
        con_ = user_cons.to(default_device())
        return con_

    def calc_gr(self, emb_u):
        origin = self.manifold.origin(args.embedding_dim)
        tmp = self.manifold.dist2(emb_u, origin, keepdim=True).clamp_max(15.0)
        gr = torch.sqrt(tmp)
        return gr

    def compute_loss(self, embeddings, triples, tag_labels, user_item_matrix):
        assert not torch.isnan(triples).any()
        triples = triples.to(default_device())
        train_edges = triples[:, [0, 1]]

        sampled_false_edges_list = [triples[:, [0, 2 + i]] for i in range(self.args.num_neg)]
        pos_scores = self.decode(embeddings, train_edges)

        neg_scores_list = [self.decode(embeddings, sampled_false_edges) for sampled_false_edges in
                           sampled_false_edges_list]
        neg_scores = torch.cat(neg_scores_list, dim=1)

        loss = pos_scores - neg_scores + self.margin

        if self.item_tag_matrix == None:
            self.item_tag_matrix = torch.from_numpy(tag_labels).float().to(default_device())
            self.item_tag_matrix = (self.item_tag_matrix > 0).float().cpu()

        if self.item_ids == None:
            self.item_ids = torch.arange(self.item_tag_matrix.shape[0])[self.item_tag_matrix.sum(dim=1) > 0]

        if self.user_item_matrix == None:
            self.user_item_matrix = torch.from_numpy(user_item_matrix).float().to(default_device())
            self.user_item_matrix = (self.user_item_matrix > 0).float().cpu()


        emb_u = embeddings[train_edges[:, 0].long()]
        alpha = self.calc_alpha(emb_u, train_edges[:, 0].long())
        loss =  loss * alpha

        loss[loss < 0] = 0
        loss = torch.sum(loss)

        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()

        if self.lam > 0:
            logits, inside_loss, disjoint_loss = self.HMI_loss(self.vtg.weight, implication, exclusion)
            loss_cla = nn.BCEWithLogitsLoss()
            tag_labels = torch.tensor(tag_labels).to(cuda_device)
            classification_loss = loss_cla(logits, tag_labels.double())
            tag_graph_loss = 1e-4 * (inside_loss + disjoint_loss)

            train_HMI_loss = self.lam * (classification_loss + tag_graph_loss)

            loss = loss + train_HMI_loss

        return loss

    def predict(self, h, data):
        num_users, num_items = data.num_users, data.num_items
        probs_matrix = np.zeros((num_users, num_items))
        for i in range(num_users):
            emb_utg = h[:, :][i].repeat(num_items).view(num_items, -1)
            emb_vtg = h[np.arange(num_users, num_users + num_items), :]
            sqdist = self.manifold.dist2(emb_utg, emb_vtg)
            probs = sqdist.detach().cpu().numpy() * -1
            probs_matrix[i] = np.reshape(probs, [-1, ])
        return probs_matrix

    def HMI_loss(self, X, implication, exclusion):

        encoded = self.ball_.projx(X)
        encoded = self.encoder_HMI(encoded)
        self.ball_.assert_check_point_on_manifold(encoded)
        log_probability = self.classifier(encoded)

        # implication
        sub_label_id = implication[:, 0]
        par_label_id = implication[:, 1]
        sub_emb_tag = self.emb_tag[sub_label_id.long()]
        par_emb_tag = self.emb_tag[par_label_id.long()]

        inside_loss = F.relu(- self.insideness(sub_emb_tag, par_emb_tag))

        # exclusion
        left_label_id = exclusion[:, 0]
        right_label_id = exclusion[:, 1]
        left_emb_tag = self.emb_tag[left_label_id.long()]
        right_emb_tag = self.emb_tag[right_label_id.long()]

        disjoint_loss = F.relu(- self.disjointedness(left_emb_tag, right_emb_tag))
        return log_probability, inside_loss.mean(), disjoint_loss.mean()

    def regularization(self, points):
        return torch.norm(torch.norm(points, p=2, dim=1, keepdim=True) - 0.5, p=2, dim=1, keepdim=True)

    def radius_regularization(self, radius):
        return torch.norm(1 - radius)

    def classifier(self, X):
        point_item = X.unsqueeze(1).expand(-1, self.emb_tag.shape[0], -1)
        point_tag = self.emb_tag.expand_as(point_item)

        logits = self.membership(point_item, point_tag, dim=2).squeeze(2)
        return logits

    def insideness(self, point_a, point_b, dim=-1):
        point_a_dist = torch.norm(point_a, p=2, dim=dim, keepdim=True)
        point_b_dist = torch.norm(point_b, p=2, dim=dim, keepdim=True)

        radius_a = (1 - point_a_dist ** 2) / (2 * point_a_dist)
        radius_b = (1 - point_b_dist ** 2) / (2 * point_b_dist)

        center_a = point_a * (1 + radius_a / point_a_dist)
        center_b = point_b * (1 + radius_b / point_b_dist)

        center_dist = torch.norm(center_a - center_b, p=2, dim=dim, keepdim=True)
        insideness = (radius_b - radius_a) - center_dist
        return insideness

    def disjointedness(self, point_a, point_b, dim=-1):
        point_a_dist = torch.norm(point_a, p=2, dim=dim, keepdim=True)
        point_b_dist = torch.norm(point_b, p=2, dim=dim, keepdim=True)

        radius_a = (1 - point_a_dist ** 2) / (2 * point_a_dist)
        radius_b = (1 - point_b_dist ** 2) / (2 * point_b_dist)

        center_a = point_a * (1 + radius_a / point_a_dist)
        center_b = point_b * (1 + radius_b / point_b_dist)

        center_dist = torch.norm(center_a - center_b, p=2, dim=dim, keepdim=True)
        disjointedness = center_dist - (radius_a + radius_b)
        return disjointedness

    def membership(self, point_a, point_b, dim=-1):
        center_a = point_a

        point_b_dist = torch.norm(point_b, p=2, dim=dim, keepdim=True)
        radius_b = (1 - point_b_dist ** 2) / (2 * point_b_dist)
        center_b = point_b * (1 + radius_b / point_b_dist)

        center_dist = torch.norm(center_a - center_b, p=2, dim=dim, keepdim=True)
        membership = radius_b - center_dist
        return membership

