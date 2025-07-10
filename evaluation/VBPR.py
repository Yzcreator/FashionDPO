import torch
import torch.nn as nn
import torch.nn.functional as F


class VBPR(nn.Module):
    def __init__(self,
                 embedding_size,
                 user_total,
                 item_total,
                 visual_feat_dim,
                 ):
        super(VBPR, self).__init__()
        self.embedding_size = embedding_size
        self.user_total = user_total
        self.item_total = item_total

        # create visual embedding for users
        self.visual_user_embeddings = nn.Embedding(user_total, embedding_size)
        self._initialize_weights(self.visual_user_embeddings)

        # create latent embedding for items
        self.latent_item_embeddings = nn.Embedding(item_total, embedding_size)
        self._initialize_weights(self.latent_item_embeddings)

        # create user and item biases
        self.user_bias = nn.Embedding(self.user_total, 1)
        self.item_bias = nn.Embedding(self.item_total, 1)
        # import ipdb; ipdb.set_trace()
        # initilization
        user_bias = torch.zeros(self.user_total, 1)
        item_bias = torch.zeros(self.item_total, 1)
        # feed values
        self.user_bias.weight.data.copy_(user_bias)
        self.item_bias.weight.data.copy_(item_bias)

        # miscs
        self.bias = nn.Parameter(torch.FloatTensor([0.0]))
        self.visual_bias = nn.Linear(visual_feat_dim, 1)

    def forward(self, u_e, i_e, vf):
    # def forward(self, u_ids, i_ids, u_e, i_e, vf):
        # batch_size = len(u_ids)
        # import ipdb; ipdb.set_trace()
        # u_ve = self.visual_user_embeddings(u_ids) # (5,64)
        # i_le = self.latent_item_embeddings(i_ids)

        # u_b = self.user_bias(u_ids)
        # i_b = self.item_bias(i_ids)

        # y = self.bias.expand(batch_size) + u_b.squeeze() + i_b.squeeze() + torch.bmm(u_e.unsqueeze(1), i_le.unsqueeze(
        #     2)).squeeze() + torch.bmm(u_ve.unsqueeze(1), i_e.unsqueeze(2)).squeeze() + self.visual_bias(vf).squeeze()

        y = torch.bmm(u_e.unsqueeze(1), i_e.unsqueeze(2)).squeeze() + self.visual_bias(vf).squeeze()
        return y

    def _initialize_weights(self , m):
        # import ipdb; ipdb.set_trace()
        if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
            nn.init.xavier_uniform_(m.weight)
            m.weight.data.copy_(F.normalize(m.weight.data, p=2, dim=1))
        elif torch.is_tensor(m):
            nn.init.xavier_uniform_(m)
            return F.normalize(m, p=2, dim=1)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)