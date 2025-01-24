import io
import scipy.io as matio
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


import os.path
import numpy as np
from PIL import Image
import time
import re
import torch
import torch.utils.data
import torch.nn.parallel as para
from torch import nn, optim
from torch.autograd import grad
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.utils.model_zoo as model_zoo
from torch.nn import Parameter
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import ipdb
import copy
from VBPR import VBPR
from torchvision import transforms
import data_utils
import random

# --Path settings----------------------------------------------------------------------------------
root_path = '/data_Allrecipes/'
result_path = 'vbpr_com/102640/'
data_path = '../datasets'
dataset_name = 'ifashion'
img_folder_path = '../datasets/ifashion/semantic_category'
load_preprocessed_features = True

if not os.path.exists(result_path):
    os.makedirs(result_path)

# --manual setting----------------------------------------------------------------------------------
# changed configuration to this instead of argparse for easier interaction
CUDA = 1  # 1 for True; 0 for False
SEED = 1
LOG_INTERVAL = 10

# user_num = 68768
# item_num = 45630
# user_num = 102642
# item_num = 102642
user_num = 102640
item_num = 102640
k = 10  # top-k items for rec
number_neg_sample_train = 1  # The proportion of negative and positive samples in the train
visual_feat_dim = 512
number_sample_eval = 500
n_fold_test_sampling = 5
i_e_mean = 0

# parameters
latent_len = 64
# BATCH_SIZE = 64
BATCH_SIZE = 10
decay = 100
EPOCHS = decay * 3 + 1
adagrad_lr = 3.4e-2
lr_decay_rate_adagrad = 0.1
opt_w_decay_rate_adagrad = 1e-5


resolution = 512
center_crop = False
random_flip = False


# --Create dataset----------------------------------------------------------------------------------
class FoodData(torch.utils.data.Dataset):
    def __init__(self):
        # load user-item interactions
        self.data_train = np.load(root_path + 'data_train.npy')[:, :2].astype(np.long)  # (676946, 2)

        # load label data
        self.labels = np.load(root_path + 'interaction_indicator_train.npy').astype(np.long)  # (68768, 45630)

    def __getitem__(self, index):
        # get data
        # load data matrix
        data = self.data_train[index]

        # get label, i.e. indicators of interacted items for the user in data
        label = self.labels[data[0]]

        return data, label

    def __len__(self):
        return len(self.data_train)




# the recommender model
class myModel(nn.Module):
    def __init__(self,
                 embedding_size,
                 user_num,
                 item_num,
                 ):
        super(myModel, self).__init__()

        # create rec model
        self.rec_model = VBPR(embedding_size, user_num, item_num, visual_feat_dim)

        # create user embeddings
        self.user_embeddings = nn.Embedding(user_num, embedding_size)  # Embedding(68768, 64)
        # initialize user embeddings
        user_weight = torch.FloatTensor(user_num, embedding_size)
        nn.init.xavier_uniform_(user_weight)
        user_weight = F.normalize(user_weight, p=2, dim=1)
        # feed values
        self.user_embeddings.weight.data.copy_(user_weight)

        # create mapping for item embedding
        self.i_map1 = nn.Linear(visual_feat_dim,
                                visual_feat_dim // 4)  # Linear(in_features=512, out_features=128, bias=True)
        nn.init.xavier_uniform_(self.i_map1.weight)
        self.i_map1.weight.data.copy_(F.normalize(self.i_map1.weight.data, p=2, dim=1))

        self.i_map2 = nn.Linear(visual_feat_dim // 4,
                                embedding_size)  # Linear(in_features=128, out_features=64, bias=True)
        nn.init.xavier_uniform_(self.i_map2.weight)
        self.i_map2.weight.data.copy_(F.normalize(self.i_map2.weight.data, p=2, dim=1))

        # misc
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, vf, uf):
        # get user/item embeddings

        i = self.leakyrelu(self.i_map1(vf))
        i_e = self.leakyrelu(self.i_map2(i)) # torch.Size([5,64])

        u = self.leakyrelu(self.i_map1(uf))
        u_e = self.leakyrelu(self.i_map2(u)) # torch.Size([5,64])

        # compute score
        y = self.rec_model(u_e, i_e, vf)

        return y


# -- lr ------------------------------------------------------------------
def lr_scheduler(epoch):
    if epoch // decay:
        decay_power = epoch // decay
        optimizer.param_groups[0]['lr'] = adagrad_lr * (lr_decay_rate_adagrad ** decay_power)

    print(
        'latent_len = {}, BATCH_SIZE = {}, decay = {}, EPOCHS = {}, lr_decay_rate_adag = {}, opt_w_decay_rate_adagrad = {}, this_adagrad_lr = {}\n'.format(
            latent_len, BATCH_SIZE, decay, EPOCHS, lr_decay_rate_adagrad, opt_w_decay_rate_adagrad,
            optimizer.param_groups[0]['lr']))

    with io.open(result_path + 'parameters.txt', 'a', encoding='utf-8') as file:
        file.write(
            'latent_len = {}, BATCH_SIZE = {}, decay = {}, EPOCHS = {}, lr_decay_rate_adag = {}, opt_w_decay_rate_adagrad = {}, this_adagrad_lr = {}\n'.format(
                latent_len, BATCH_SIZE, decay, EPOCHS, lr_decay_rate_adagrad, opt_w_decay_rate_adagrad,
                optimizer.param_groups[0]['lr']))
    return optimizer.param_groups[0]['lr']


# -- training ------------------------------------------------------------------
def train(epoch):
    print('Training starts..')
    # toggle model to train mode
    model_rec.train()

    # initialize loss
    all_loss = 0

    total_time = time.time()
    for batch_idx, data in enumerate(train_dataloader):
        # data:torch.Size([64, 2])
        # label:torch.Size([64, 45630])
        start_time = time.time()

        # make input data
        user_ids = data['outfit_ids'] # Tensor:(bsz:10,)
        user_ids = user_ids.repeat(2)  # Tensor:(20,)

        pos_ids = data['pos_samples_ids'] # Tensor:(bsz:10,)
        neg_ids = data['neg_samples_ids']# Tensor:(bsz:10,)
        sample_ids = torch.cat([pos_ids, neg_ids]) # Tensor:(20,)

        pos_samples = data['encode_pos_samples'] # Tensor:(10, 512) # 正样本id
        neg_samples = data['encode_neg_samples'] # Tensor:(10, 512) # 负样本id
        item_feats = torch.cat([pos_samples, neg_samples]) # Tensor:(20, 512)

        user_samples = data['encode_in_outfits_samples'] # Tensor:(batchsize:10, out_features:512) # 用户id
        user_feats = torch.cat([user_samples,user_samples]) # (20, 512)

        if CUDA:
            user_ids = user_ids.cuda()
            sample_ids = sample_ids.cuda()
            item_feats = item_feats.cuda()
            user_feats = user_feats.cuda()

        scores_true = model_rec(item_feats, user_feats)
        loss = loss_function(scores_true[:len(pos_samples)], scores_true[len(pos_samples):])

        all_loss += loss.item()
        # optim for model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1 and batch_idx == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)] | Loss_all: {:.4f} | Time:{} | Total_Time:{}\n'.format(epoch, (
                            batch_idx + 1) * len(data), len(train_dataloader.dataset), 100. * (batch_idx + 1) / len(
                    train_dataloader), loss, round((time.time() - start_time), 4), round((time.time() - total_time), 4)))

            with io.open(result_path + 'train_loss.txt', 'a', encoding='utf-8') as file:
                file.write('	First batch: loss_all:{}\n'.format(loss))

        elif batch_idx % LOG_INTERVAL == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)] | Loss: {:.4f} | Time:{} | Total_Time:{}\n'.format(
                    epoch, (batch_idx + 1) * len(data), len(train_dataloader.dataset),
                           100. * (batch_idx + 1) / len(train_dataloader), loss, round((time.time() - start_time), 4),
                    round((time.time() - total_time), 4)))

            # records current progress for tracking purpose
            with io.open(result_path + 'model_batch_true_loss.txt', 'a', encoding='utf-8') as file:
                file.write(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] | Loss: {:.4f} | Time:{} | Total_Time:{}\n'.format(
                        epoch, (batch_idx + 1) * len(data), len(train_dataloader.dataset),
                               100. * (batch_idx + 1) / len(train_dataloader), loss,
                               round((time.time() - start_time), 4) * LOG_INTERVAL,
                        round((time.time() - total_time), 4)))

    print(
        '====> Epoch: {} | Average loss_all: {:.4f} | Time:{}'.format(
            epoch, all_loss / len(train_dataloader), round((time.time() - total_time), 4)))

    with io.open(result_path + 'train_loss.txt', 'a', encoding='utf-8') as file:
        file.write('	Epoach {}: loss_all:{}\n'.format(epoch, all_loss / len(train_dataloader)))
    return all_loss / len(train_dataloader)


# -- prepare bpr samples ------------------------------------------------------------
# def prepare_sample(data, labels, number_neg_sample_train, valid_item_entries_train):
#     # get user/item ids
#     user_ids = data[:, 0]
#     item_ids = data[:, 1]
#
#     # sample a negative item for each user
#     neg_samples = []
#     for i in range(len(user_ids)):
#         # get the list of negative samples
#         # get positive samples of the user
#         pos_list = np.where(labels[i] > 0)[0]
#         # remove pos samples from train items
#
#         while True:
#             # sample an item_id
#             neg_item_index = np.random.randint(0, len(valid_item_entries_train), size=number_neg_sample_train)
#             neg_item_id = valid_item_entries_train[neg_item_index]
#             # check existance
#             if not (neg_item_id in pos_list):
#                 neg_samples.append(neg_item_id)
#                 break
#
#     return user_ids, item_ids, torch.tensor(neg_samples).squeeze(1)  # torch.from_numpy(np.stack(neg_samples))


# -- Loss ------------------------------------------------------------------
def loss_function(pos_scores, neg_scores):
    bpr_loss = - F.logsigmoid(pos_scores - neg_scores)
    return torch.mean(bpr_loss)


# -- optimizer -------------------------------------------------------------
def get_optim():
    # params in ingre prediction net
    optimizer = optim.Adagrad(model_rec.parameters(), lr=adagrad_lr, weight_decay=opt_w_decay_rate_adagrad)
    return optimizer


# -- test ------------------------------------------------------------------
def test(user_list_test_cold, user_list_test_dense, user_list_test, interaction_indicator_test,
         interaction_indicator_train, valid_item_entries_train):
    print('Start test...')
    # toggle to eval mode
    model_rec.eval()

    # create for avg test loss
    # create for avg test loss
    p_cold = np.zeros(n_fold_test_sampling)
    r_cold = np.zeros(n_fold_test_sampling)
    f_cold = np.zeros(n_fold_test_sampling)
    ndcg_cold = np.zeros(n_fold_test_sampling)
    auc_cold = np.zeros(n_fold_test_sampling)
    p_dense = np.zeros(n_fold_test_sampling)
    r_dense = np.zeros(n_fold_test_sampling)
    f_dense = np.zeros(n_fold_test_sampling)
    ndcg_dense = np.zeros(n_fold_test_sampling)
    auc_dense = np.zeros(n_fold_test_sampling)
    p_total = np.zeros(n_fold_test_sampling)
    r_total = np.zeros(n_fold_test_sampling)
    f_total = np.zeros(n_fold_test_sampling)
    ndcg_total = np.zeros(n_fold_test_sampling)
    auc_total = np.zeros(n_fold_test_sampling)
    # compute the performance for each user in test
    count = 0
    for user in user_list_test_cold:
        if not (count % 5000):
            print('process test user {}'.format(count))
        count += 1
        # create item input
        gt_item_ids, item_ids = sampler(False, user, interaction_indicator_test, interaction_indicator_train,
                                        valid_item_entries_train, number_sample_eval)

        # get item feats
        item_feats = imgs_feats[item_ids.numpy()]  # torch.Size([128, 512])

        # create the user input
        user_ids = torch.from_numpy(np.array(user)).unsqueeze(0)
        user_ids = user_ids.repeat(len(item_ids))

        if CUDA:
            gt_item_ids = gt_item_ids.cuda()
            user_ids = user_ids.cuda()
            item_ids = item_ids.cuda()
            item_feats = item_feats.cuda()

        # compute relavant scores
        scores = model_rec(user_ids, item_ids, item_feats)

        # compute performance
        for i in range(n_fold_test_sampling):
            # partition data
            result_batch = torch.cat([
                scores[:len(gt_item_ids)],
                scores[len(gt_item_ids):][i * number_sample_eval:(i + 1) * number_sample_eval]])  # scores of gt_item_ids and neg_item_ids in this fold
            p, r, f, ndcg, auc = top_match(result_batch, torch.cat(
                [gt_item_ids, item_ids[len(gt_item_ids):][i * number_sample_eval:(i + 1) * number_sample_eval]]),
                                           interaction_indicator_test[user], k)
            p_cold[i] += p
            r_cold[i] += r
            f_cold[i] += f
            ndcg_cold[i] += ndcg
            auc_cold[i] += auc

    print('cold items processed!')
    for user in user_list_test_dense:
        if not (count % 5000):
            print('process test user {}'.format(count))
        count += 1
        # create item input
        gt_item_ids, item_ids = sampler(False, user, interaction_indicator_test, interaction_indicator_train,
                                        valid_item_entries_train, number_sample_eval)

        # get item feats
        item_feats = imgs_feats[item_ids.numpy()]  # torch.Size([128, 512])

        # create the user input
        user_ids = torch.from_numpy(np.array(user)).unsqueeze(0)
        user_ids = user_ids.repeat(len(item_ids))

        if CUDA:
            gt_item_ids = gt_item_ids.cuda()
            user_ids = user_ids.cuda()
            item_ids = item_ids.cuda()
            item_feats = item_feats.cuda()

        # compute relavant scores
        scores = model_rec(user_ids, item_ids, item_feats)

        # compute performance
        for i in range(n_fold_test_sampling):
            # partition data
            result_batch = torch.cat([
                scores[:len(gt_item_ids)],
                scores[len(gt_item_ids):][i * number_sample_eval:(
                                                                             i + 1) * number_sample_eval]])  # scores of gt_item_ids and neg_item_ids in this fold
            p, r, f, ndcg, auc = top_match(result_batch, torch.cat(
                [gt_item_ids, item_ids[len(gt_item_ids):][i * number_sample_eval:(i + 1) * number_sample_eval]]),
                                           interaction_indicator_test[user], k)
            p_dense[i] += p
            r_dense[i] += r
            f_dense[i] += f
            ndcg_dense[i] += ndcg
            auc_dense[i] += auc
    print('dense items processed!')
    p_total = p_cold + p_dense
    r_total = r_cold + r_dense
    f_total = f_cold + f_dense
    ndcg_total = ndcg_cold + ndcg_dense
    auc_total = auc_cold + auc_dense

    # get mean, std of performance
    print("For the cold subset:")
    p1 = get_statistics(p_cold / len(user_list_test_cold))  # the mean,std,max,min of p
    r1 = get_statistics(r_cold / len(user_list_test_cold))  # the mean,std,max,min of r
    f1 = get_statistics(f_cold / len(user_list_test_cold))  # the mean,std,max,min of f
    ndcg1 = get_statistics(ndcg_cold / len(user_list_test_cold))  # the mean,std,max,min of ndcg
    auc1 = get_statistics(auc_cold / len(user_list_test_cold))  # the mean,std,max,min of auc
    print('mean: {},{},{},{},{}\n'.format(p1[0], r1[0], f1[0], ndcg1[0], auc1[0]))
    print('std: {},{},{},{},{}\n'.format(p1[1], r1[1], f1[1], ndcg1[1], auc1[1]))
    print('max: {},{},{},{},{}\n'.format(p1[2], r1[2], f1[2], ndcg1[2], auc1[2]))
    print('min: {},{},{},{},{}\n'.format(p1[3], r1[3], f1[3], ndcg1[3], auc1[3]))

    print("For the dense subset:")
    p2 = get_statistics(p_dense / len(user_list_test_dense))  # the mean,std,max,min of p
    r2 = get_statistics(r_dense / len(user_list_test_dense))  # the mean,std,max,min of r
    f2 = get_statistics(f_dense / len(user_list_test_dense))  # the mean,std,max,min of f
    ndcg2 = get_statistics(ndcg_dense / len(user_list_test_dense))  # the mean,std,max,min of ndcg
    auc2 = get_statistics(auc_dense / len(user_list_test_dense))  # the mean,std,max,min of auc
    print('mean: {},{},{},{},{}\n'.format(p2[0], r2[0], f2[0], ndcg2[0], auc2[0]))
    print('std: {},{},{},{},{}\n'.format(p2[1], r2[1], f2[1], ndcg2[1], auc2[1]))
    print('max: {},{},{},{},{}\n'.format(p2[2], r2[2], f2[2], ndcg2[2], auc2[2]))
    print('min: {},{},{},{},{}\n'.format(p2[3], r2[3], f2[3], ndcg2[3], auc2[3]))

    print("For the total subset:")
    p3 = get_statistics(p_total / len(user_list_test))  # the mean,std,max,min of p
    r3 = get_statistics(r_total / len(user_list_test))  # the mean,std,max,min of r
    f3 = get_statistics(f_total / len(user_list_test))  # the mean,std,max,min of f
    ndcg3 = get_statistics(ndcg_total / len(user_list_test))  # the mean,std,max,min of ndcg
    auc3 = get_statistics(auc_total / len(user_list_test))  # the mean,std,max,min of auc
    print('mean: {},{},{},{},{}\n'.format(p3[0], r3[0], f3[0], ndcg3[0], auc3[0]))
    print('std: {},{},{},{},{}\n'.format(p3[1], r3[1], f3[1], ndcg3[1], auc3[1]))
    print('max: {},{},{},{},{}\n'.format(p3[2], r3[2], f3[2], ndcg3[2], auc3[2]))
    print('min: {},{},{},{},{}\n'.format(p3[3], r3[3], f3[3], ndcg3[3], auc3[3]))

    return p1, r1, f1, ndcg1, auc1, p2, r2, f2, ndcg2, auc3, p3, r3, f3, ndcg3, auc3


# -- sampler ------------------------------------------------------------------
def sampler(val, user_id, interaction_indicator_test, interaction_indicator_train, valid_item_entries_train,
            number_sample_eval):
    # get ground-truth items
    gt_item_ids = np.where(interaction_indicator_test[user_id] > 0)[0]

    # get negative samples
    # filter samples
    if val:  # if val, also remove pos samples in val for true response to neg samples
        samples_to_filter = np.concatenate([gt_item_ids, np.where(interaction_indicator_train[user_id] > 0)[0]])
    else:  # if test
        gt_item_ids_train = np.where(interaction_indicator_train[user_id] > 0)[0]
        samples_to_filter = np.array(list(gt_item_ids_train) + list(gt_item_ids))
    sample_pool = np.setdiff1d(valid_item_entries_train, samples_to_filter)
    # sample neg samples from the pool
    sample_list = []
    for i in range(n_fold_test_sampling):
        neg_sample_ids = np.random.choice(sample_pool, number_sample_eval, replace=False)
        sample_list.append(neg_sample_ids)

    sample_list = np.concatenate(sample_list)
    # concate gt and neg samples
    sample_ids = np.concatenate([gt_item_ids, sample_list])

    return torch.from_numpy(gt_item_ids), torch.from_numpy(sample_ids)


def top_match(rec_scores, item_ids, true_indicator, k):
    # get rec item ids
    rec_scores = rec_scores.detach().cpu().numpy()
    rank_list = (-rec_scores).argsort()[:k]  # start from highest
    rec_list = item_ids[rank_list]

    # get true item ids
    true_list = np.where(true_indicator > 0)[0]

    # compute top-k hits
    hit = 0
    hit_indicator = []
    for item_id in rec_list:
        if item_id in true_list:
            hit += 1
            hit_indicator.append(1)
        else:
            hit_indicator.append(0)

    # compute performance
    f = 0.0
    p = 0.0
    r = 0.0
    ndcg = 0.0
    if hit:
        p = float(hit) / k
        r = float(hit) / len(true_list)
        f = 2 * p * r / (p + r)
        ndcg = ndcg_at_k(hit_indicator, k)
    auc = compute_auc(rec_scores)

    return p, r, f, ndcg, auc


# calculate the mean,std,max and min
def get_statistics(values):
    mean = np.mean(values)
    std = np.std(values)
    pmax = np.max(values)
    pmin = np.min(values)
    return [mean, std, pmax, pmin]


# -- performance compute -------------------------------------------------------------
# calculate the ndcg
def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

    # calculate the dcg


def dcg_at_k(r, k, method=1):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

    # calculate the auc


def compute_auc(scores):
    num_pos = len(scores) - number_sample_eval
    score_neg = scores[num_pos:]
    num_hit = 0

    for i in range(num_pos):
        num_hit += len(np.where(score_neg < scores[i])[0])

    auc = num_hit / (num_pos * number_sample_eval)
    return auc


def get_Cold_and_Dense_user_list(interaction_indicator_test):
    user_list_test_cold = []
    user_list_test_dense = []
    for i in range(interaction_indicator_test.shape[0]):
        if np.sum(interaction_indicator_test[i]) <= 2:
            user_list_test_cold.append(i)
        else:
            user_list_test_dense.append(i)
    print(len(user_list_test_cold))
    print(len(user_list_test_dense))
    return user_list_test_cold, user_list_test_dense


# --data loader----------------------------------------------------------------------------------
print('save in ' + result_path)
torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)
# DataLoader instances
kwargs = {'num_workers': 0, 'pin_memory': True} if CUDA else {}

print("Data loading......")
data_path = os.path.join(data_path, dataset_name)
train_dict = np.load(os.path.join(data_path, "processed", "train.npy"), allow_pickle=True).item()
# train_history = np.load(os.path.join(data_path, "processed", "train_hist_latents.npy"), allow_pickle=True).item()
# train_grd_dict = np.load(os.path.join(data_path, "train.npy"), allow_pickle=True).item()
# train_grd_dict = np.load(os.path.join(data_path, "test_grd_dict.npy"), allow_pickle=True).item()


new_id_cate_dict = np.load(os.path.join(data_path, "id_cate_dict.npy"), allow_pickle=True).item()
all_image_paths = np.load(os.path.join(data_path, "all_item_image_paths.npy"), allow_pickle=True)

img_trans = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor()
        ]
    )
img_dataset = data_utils.ImagePathDataset(img_folder_path, all_image_paths, img_trans, do_normalize=True)

train_data_dict = train_dict




# train_loader = torch.utils.data.DataLoader(
#     FoodData(), batch_size=BATCH_SIZE, shuffle=True, **kwargs)

# # load interaction indicator of train data
# interaction_indicator_train = np.load(root_path + 'interaction_indicator_train.npy')  # (68768, 45630)
# # compute popularity for train samples
# item_freq_counts_train = np.sum(interaction_indicator_train, 0)  # (45630,)
# # get entries of non-empty items
# valid_item_entries_train = np.where(item_freq_counts_train > 0)[0]  # (29093,)
# # load pretrained img feats
# imgs_feats = torch.from_numpy(np.load(root_path + 'img_feats_pretrain_ingre_resnet18.npy'))  # (45630, 512)
# print('train data prepared...')
#
# # load interaction indicator of test data
# # load interaction indicators
# interaction_indicator_test = np.load(root_path + 'interaction_indicator_test.npy')  # (68768, 45630)
# # load user dict
# user_list_test = np.load(root_path + 'user_list_test.npy')  # (68768,)
# print('test data prepared...')


# data preprocessing  /   get image feature

## load resnet model
resnet18 = models.resnet18(pretrained=False)
resnet18.load_state_dict(torch.load('pretrained-models/resnet18/resnet18-f37072fd.pth'))
resnet18.fc = nn.Linear(resnet18.fc.in_features, 512)
resnet18.eval()

# preprocess = transforms.Compose([
#     transforms.Resize((512, 512)),  # 调整图像大小
#     transforms.ToTensor(),  # 转换为张量
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
# ])
#
# def load_and_preprocess_image(image_path):
#     img = Image.open(image_path).convert('RGB')  # 加载图像并转换为RGB
#     img_tensor = preprocess(img)  # 预处理图像
#     return img_tensor

def encode_image(img_tensor, model):
    img_tensor = img_tensor.unsqueeze(0)
    with torch.no_grad():
        encoded_img = model(img_tensor)
    return encoded_img.squeeze(0)

## prepare dataset
encode_pos_samples = []
encode_neg_samples = []
encode_in_outfits_samples = []
pos_samples_ids = []
neg_samples_ids = []
outfit_ids = []
in_outfits_ids = []
if load_preprocessed_features == False:
    oids = train_data_dict['oids']
    outfits = train_data_dict['outfits']
    for i, oid in enumerate(oids):
        blank = 0
        incomplete_outfits = outfits[i].tolist()
        # if oid in train_grd_dict['oids']:
        #     index = train_grd_dict['oids'].index(oid)
        #     grd_outfits = train_grd_dict['outfits'][index]

        choose = random.randint(0, 4)
        pos_samples_id = 0
        neg_samples_id = random.randint(1, 10000)
        incomplete_outfits_id = []
        for i in range(len(incomplete_outfits)):
            if i == choose:
                pos_samples_id = incomplete_outfits[i]
            else:
                incomplete_outfits_id.append(incomplete_outfits[i])

        in_outfits_ids.append(incomplete_outfits_id)
        pos_samples_ids.append(pos_samples_id)
        neg_samples_ids.append(neg_samples_id)
        outfit_ids.append(oid)

    ## get image feature
    for i, pos_id in enumerate(pos_samples_ids):
        pos_image = img_dataset[pos_id]
        encoded_pos_image = encode_image(pos_image, resnet18)
        encode_pos_samples.append(encoded_pos_image)

        neg_image = img_dataset[neg_samples_ids[i]]
        encoded_neg_image = encode_image(neg_image, resnet18)
        encode_neg_samples.append(encoded_neg_image)

        encode_in_outfits_images = []
        for j, in_outfit_id in enumerate(in_outfits_ids[i]):
            in_outfits_image = img_dataset[in_outfit_id]
            encoded_in_outfits_image = encode_image(in_outfits_image, resnet18)
            encode_in_outfits_images.append(encoded_in_outfits_image)
        # encode_in_outfits_samples.append(torch.cat(encode_in_outfits_images, dim=0)) # cat
        average_encode_in_outfits_images = torch.mean(torch.stack(encode_in_outfits_images), dim=0)  # average
        encode_in_outfits_samples.append(average_encode_in_outfits_images)

    torch.save(encode_pos_samples, f'vbpr_com/{item_num}/encode_pos_samples_{item_num}.pt')
    torch.save(encode_neg_samples, f'vbpr_com/{item_num}/encode_neg_samples_{item_num}.pt')
    torch.save(encode_in_outfits_samples, f'vbpr_com/{item_num}/encode_in_outfits_samples_{item_num}.pt')
    torch.save(pos_samples_ids, f'vbpr_com/{item_num}/pos_samples_ids_{item_num}.pt')
    torch.save(neg_samples_ids, f'vbpr_com/{item_num}/neg_samples_ids_{item_num}.pt')
    torch.save(outfit_ids, f'vbpr_com/{item_num}/outfit_ids_{item_num}.pt')


else:
    encode_pos_samples = torch.load(f'vbpr_com/{item_num}/encode_pos_samples_{item_num}.pt')
    encode_neg_samples = torch.load(f'vbpr_com/{item_num}/encode_neg_samples_{item_num}.pt')
    encode_in_outfits_samples = torch.load(f'vbpr_com/{item_num}/encode_in_outfits_samples_{item_num}.pt')
    pos_samples_ids = torch.load(f'vbpr_com/{item_num}/pos_samples_ids_{item_num}.pt')
    neg_samples_ids = torch.load(f'vbpr_com/{item_num}/neg_samples_ids_{item_num}.pt')
    outfit_ids = torch.load(f'vbpr_com/{item_num}/outfit_ids_{item_num}.pt')

train_data_dict['encode_pos_samples'] = encode_pos_samples
train_data_dict['encode_neg_samples'] = encode_neg_samples
train_data_dict['encode_in_outfits_samples'] = encode_in_outfits_samples
train_data_dict['pos_samples_ids'] = pos_samples_ids
train_data_dict['neg_samples_ids'] = neg_samples_ids
train_data_dict['outfit_ids'] = outfit_ids



train_dataset = data_utils.FashionDiffusionData(train_data_dict)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=BATCH_SIZE,
    num_workers=0,
)
print('train data finish prepared.')


model_rec = myModel(latent_len, user_num, item_num)
print('model created...')

model_rec = torch.nn.DataParallel(model_rec)
if CUDA:
    model_rec = model_rec.cuda()

# optimizer
optimizer = get_optim()
print('optimizer prepared...')

index_best_epoch = 10000
loss_best_epoch = 10000
loss_epoch = 0

# user_list_test_cold, user_list_test_dense = get_Cold_and_Dense_user_list(interaction_indicator_test)

# -- training process ----------------------------------------------------------------
for epoch in range(1, EPOCHS + 1):
    this_lr = lr_scheduler(epoch)
    # train
    loss_epoch = train(epoch)
    torch.save(model_rec.state_dict(), result_path + 'model_rec-{}.pt'.format(epoch))
    if loss_epoch <= loss_best_epoch:
        loss_best_epoch = loss_epoch
        index_best_epoch = epoch
        

with io.open(result_path + 'All_test_performance.txt', 'a', encoding='utf-8') as file:
    file.write('\n')
    file.write('\n')
    file.write('the best epoch is {}\n'.format(index_best_epoch))
    file.write('the best loss is {}\n'.format(loss_best_epoch))
