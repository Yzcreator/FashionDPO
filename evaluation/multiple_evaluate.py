import os
import ast
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser



import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import open_clip
import requests
from PIL import Image
from torch import nn, optim
import re
from transformers import AutoModel, AutoTokenizer
import time
import data_utils
import torchvision.models as models
from VBPR import VBPR
from torch.nn import functional as F
from tqdm import tqdm


try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

import eval_utils

os.environ['CURL_CA_BUNDLE'] = ''
os.environ['HTTP_PROXY'] = "socks5://127.0.0.1:10808"
os.environ['HTTPS_PROXY'] = "socks5://127.0.0.1:10808"
os.environ['ALL_PROXY'] = "socks5://127.0.0.1:10808"

resolution = 512
center_crop = False
random_flip = False
visual_feat_dim = 512

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
item_num = 50000



parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--continue_flag', type=str, default=False)
parser.add_argument('--evanumber', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1,
                    help='Batch size to use')
parser.add_argument('--vbpr_checkpoint_path', type=str, default=f'vbpr_com/{item_num}')
parser.add_argument('--num_workers', type=int, default=1,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--gpu', type=int, default='3',
                    help='gpu id to use')
parser.add_argument('--dims4fid', type=int, default=2048,
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('--data_path', type=str, default='../datasets/ifashion')
parser.add_argument('--img_folder_path', type=str, default='../datasets/ifashion/semantic_category')
parser.add_argument('--pretrained_evaluator_ckpt', type=str,
                    default='./compatibility_evaluator/ifashion-ckpt/fashion_evaluator.pth')
parser.add_argument('--dataset', type=str, default="ifashion")
parser.add_argument('--output_dir', type=str, default="../FashionDPO/output/sample_ddim/sample_5_ifashion_7_1000")
parser.add_argument('--eval_version', type=str, default="sample_5_ifashion_7_1000")
parser.add_argument('--ckpt', type=int, default=15000)
parser.add_argument('--num_per_outfit', type=int, default=7)
# parser.add_argument('--ckpts', type=str, default=None)
parser.add_argument('--ckpts', type=str, default="all")
parser.add_argument('--task', type=str, default="FITB")
parser.add_argument('--num_classes', type=int, default=50)
parser.add_argument('--sim_func', type=str, default="cosine")
parser.add_argument('--lpips_net', type=str, default="vgg")
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--log_name', type=str, default="log")
# parser.add_argument('--mode', type=str, default="valid")
parser.add_argument('--mode', type=str, default="test")
parser.add_argument('--hist_scales', type=float, default=4.0)
parser.add_argument('--mutual_scales', type=float, default=5.0)
parser.add_argument('--cate_scales', type=float, default=12.0)

SPECIAL_CATES = ["shoes", "pants", "sneakers", "boots", "earrings", "slippers", "sandals"]
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        u_e = u_e.unsqueeze(0)
        i_e = i_e.unsqueeze(0)
        vf = vf.unsqueeze(0)
        y = self.rec_model(u_e, i_e, vf)

        return y


class FashionEvalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class FashionRetrievalDataset(Dataset):
    def __init__(self, gen_images, candidates):
        self.gen_images = gen_images
        self.candidates = candidates

    def __len__(self):
        return len(self.gen_images)

    def __getitem__(self, index):
        return self.gen_images[index], self.candidates[index]



class FashionPersonalSimDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data["gen"])

    def __getitem__(self, index):
        gen = self.data["gen"][index]
        hist = self.data["hist"][index]

        return gen, hist

class FashionCPMquaSimDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data["image"])

    def __getitem__(self, index):
        image = self.data["image"][index]
        prompt = self.data["prompt"][index]

        return image, prompt

class FashionCPMcomSimDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data["image"])

    def __getitem__(self, index):
        image = self.data["image"][index]
        prompt = self.data["prompt"][index]

        return image, prompt


def cate_trans(cid, id_cate_dict):
    def contains_any_special_cate(category, special_cates):
        for special_cate in special_cates:
            if special_cate in category:
                return True
        return False

    category = id_cate_dict[cid]
    if contains_any_special_cate(category, SPECIAL_CATES):
        prompt = "There is a photo of a pair of " + category + ", on white background"
    else:
        prompt = "There is a photo of a " + category + ", on white background"

    return prompt


def extract_score5(response):
    quality_standards = {
        "Poor Quality": 1,
        "Low Quality": 2,
        "Moderate Quality": 3,
        "High Quality": 4,
        "Exceptional Quality": 5,
        "Incompatible": 1,
        "Slightly Compatible": 2,
        "Moderately Compatible": 3,
        "Very Compatible": 4,
        "Perfectly Compatible": 5
    }
    pattern = '|'.join(re.escape(key) for key in quality_standards.keys())

    match = re.search(r'\b([1-5])-([A-Za-z]*)', response)
    if match:
        return match.group(1)

    match = re.search(r'\b([1-5])\b', response)
    if match:
        return match.group(1)

    match = re.search(pattern, response)
    if match:
        return str(quality_standards[match.group(0)])

    return "Score not found"

def extract_score10(response):
    quality_standards = {
        "Very Poor Quality": 1,
        "Poor Quality": 2,
        "Low Quality": 3,
        "Below Average Quality": 4,
        "Moderate Quality": 5,
        "Above Average Quality": 6,
        "Good Quality": 7,
        "Very Good Quality": 8,
        "High Quality": 9,
        "Exceptional Quality": 10,
        "Very Incompatible": 1,
        "Incompatible": 2,
        "Slightly Incompatible": 3,
        "Slightly Compatible": 4,
        "Moderately Compatible": 5,
        "Fairly Compatible": 6,
        "Compatible": 7,
        "Very Compatible": 8,
        "Highly Compatible": 9,
        "Perfectly Compatible": 10,

    }
    pattern = '|'.join(re.escape(key) for key in quality_standards.keys())

    match = re.search(r'\b([1-10])-([A-Za-z]*)', response)
    if match:
        return match.group(1)

    match = re.search(r'\b([1-10])\b', response)
    if match:
        return match.group(1)

    match = re.search(pattern, response)
    if match:
        return str(quality_standards[match.group(0)])

    return "Score not found"

def encode_image(img_tensor, model):
    img_tensor = img_tensor.unsqueeze(0)
    with torch.no_grad():
        encoded_img = model(img_tensor)
    return encoded_img.squeeze(0)



def main():
    args = parser.parse_args()
    set_random_seed(args.seed)

    if args.dataset == "ifashion":
        args.data_path = '../datasets/ifashion'
        args.pretrained_evaluator_ckpt = './compatibility_evaluator/ifashion-ckpt/ifashion_evaluator.pth'
        args.output_dir = '../FashionDPO/output/sample_ddim'
    elif args.dataset == "polyvore":
        args.data_path = '../datasets/polyvore'
        args.pretrained_evaluator_ckpt = './compatibility_evaluator/polyvore-ckpt/polyvore_evaluator.pth'
        args.output_dir = '../FashionDPO/output/sample_ddim'
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}.")


    eval_path = os.path.join(args.output_dir, args.eval_version, "eval-test-git")

    # scale = f"cate{args.cate_scales}"
    scale = f"cate{args.cate_scales}-mutual{args.mutual_scales}-hist{args.hist_scales}"

    if args.gpu is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(f"cuda:{args.gpu}")
    print(f"Evaluate on device {device}")

    num_workers = args.num_workers


    # id_cate_dict = np.load(os.path.join(args.data_path, "new_id_cate_dict.npy"), allow_pickle=True).item()
    id_cate_dict = np.load(os.path.join(args.data_path, "id_cate_dict.npy"), allow_pickle=True).item()
    cid_to_label = np.load('./finetuned_inception/cid_to_label.npy',
                           allow_pickle=True).item()  # map cid to inception predicted label
    cnn_features_clip = np.load(os.path.join(args.data_path, "cnn_features_clip.npy"), allow_pickle=True)
    cnn_features_clip = torch.tensor(cnn_features_clip)


    history = np.load(os.path.join(args.data_path, "processed", "sample_history_clipembs.npy"),
                      allow_pickle=True).item()
    fitb_retrieval_candidates = np.load(os.path.join(args.data_path, "fitb_sample_retrieval_candidates.npy"),
                                        allow_pickle=True).item()
    fitb_dict = np.load(os.path.join(args.data_path, "fitb_sample_dict.npy"), allow_pickle=True).item()



    eval_save_path = os.path.join(eval_path, f"eval_results_{args.evanumber}.npy")
    eval_save_path_qua = os.path.join(eval_path, f"Intermediate_results_qua_{args.evanumber}.npy")
    eval_save_path_com = os.path.join(eval_path, f"Intermediate_results_com_{args.evanumber}.npy")
    # txt_path = os.path.join(eval_path, f"response_{args.evanumber}.txt")
    # if not os.path.exists(txt_path):
    #     with open("output.txt", "w") as file:
    #         file.write("")
    # print(f"save_path:{eval_save_path}")

    all_image_paths = np.load(os.path.join(args.data_path, "all_item_image_paths.npy"), allow_pickle=True)
    img_trans = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor()
        ]
    )
    img_dataset = data_utils.ImagePathDataset(args.img_folder_path, all_image_paths, img_trans, do_normalize=True)



    if not os.path.exists(eval_save_path):
        all_eval_metrics = {}
    else:
        all_eval_metrics = np.load(eval_save_path, allow_pickle=True).item()

    ckpt = args.ckpt
    if not args.continue_flag:
        gen_data = np.load(os.path.join(eval_path, f"{args.task}-checkpoint-{ckpt}-{scale}.npy"),
                       allow_pickle=True).item()
    else:
        gen_data = np.load(os.path.join(eval_path, f"eval_results_{args.evanumber}.npy"), allow_pickle=True).item()

    grd_data = np.load(os.path.join(eval_path, f"{args.task}-grd-new.npy"), allow_pickle=True).item()

    trans = transforms.ToTensor()
    resize = transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR)
    _, _, img_trans = open_clip.create_model_and_transforms('ViT-H-14', pretrained='../FashionDPO/models/huggingface/open_clip/open_clip_pytorch_model.bin')


    print("Evaluating Start!")


    # -------------------------------------------------------------- #
    #             get each image's compatibility score             #
    # -------------------------------------------------------------- #
    if not args.continue_flag:
        resnet18 = models.resnet18(pretrained=False)
        resnet18.load_state_dict(torch.load('pretrained-models/resnet18/resnet18-f37072fd.pth'))
        resnet18.fc = nn.Linear(resnet18.fc.in_features, 512)
        resnet18.eval()
        resnet18 = resnet18.to(device)

        outfits = []
        gen_imgs = []
        incomple_imgs = []
        gen_num = 0

        process = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

        for uid in gen_data:
            for oid in gen_data[uid]:
                outfit = fitb_dict[uid][oid]  # 0 as blank to be filled
                # real iid is positive, generated iid is negative
                for _ in range(args.num_per_outfit):
                    new_outfit = []
                    for iid in outfit:
                        if iid == 0:
                            new_outfit.append(-gen_num)
                        else:
                            new_outfit.append(iid)
                    gen_num += 1
                    outfits.append(new_outfit)

                for img_path in gen_data[uid][oid]["image_paths"]:
                    if isinstance(img_path, list):
                        gen_img_path = img_path[0]
                        gen_img_path = os.path.join(f"../FashionDPO/", gen_img_path)
                    else:
                        gen_img_path = os.path.join(f"../FashionDPO/", img_path)
                    # gen_img_path = os.path.join(f"../FashionDPO/", gen_img_path)
                    im = Image.open(gen_img_path)
                    im_tensor = process(im)
                    im_tensor = im_tensor.to(device)
                    encode_im = encode_image(im_tensor, resnet18)
                    gen_imgs.append(encode_im)

                convs = []
                incomplete_outfit = gen_data[uid][oid]["outfits"]
                for i in range(len(incomplete_outfit)):
                    if incomplete_outfit[i] == 0:
                        continue
                    else:
                        incomplete_im = img_dataset[incomplete_outfit[i]]
                        incomplete_im = incomplete_im.to(device)
                        encoded_incomplete_image = encode_image(incomplete_im, resnet18)
                        convs.append(encoded_incomplete_image)
                average_encode_in_outfits_images = torch.mean(torch.stack(convs), dim=0)
                for _ in range(args.num_per_outfit):
                    incomple_imgs.append(average_encode_in_outfits_images)

        torch.save(gen_imgs, os.path.join(args.output_dir, args.eval_version, "gen_imgs.pt"))
        torch.save(incomple_imgs, os.path.join(args.output_dir, args.eval_version, "incomple_imgs.pt"))

        # VBPR
        model_rec = myModel(latent_len, item_num, item_num)
        model_rec = torch.nn.DataParallel(model_rec, device_ids = [args.gpu])
        print('VBPR model created...')
        path_VBPR = os.path.join(args.vbpr_checkpoint_path, "model_rec-260.pt")
        state_dict = torch.load(path_VBPR)
        model_rec.load_state_dict(state_dict)
        model_rec = model_rec.to(device)
        model_rec.eval()

        com_scores = []
        for i, item_feats in enumerate(tqdm(gen_imgs, desc="Processing")):
            user_feats = incomple_imgs[i]
            user_feats = user_feats.to(device)
            item_feats = item_feats.to(device)
            scores = model_rec(item_feats, user_feats)
            scores = scores.item()
            com_scores.append(scores)

        # compatibility_score = eval_utils.evaluate_compatibility_VBPR(
        #     outfit_dataset,
        #     grd_outfit_dataset,
        #     gen_imgs,
        #     cnn_feat_path,
        #     cnn_feat_gen_path,
        #     args.pretrained_evaluator_ckpt,
        #     batch_size=args.batch_size,
        #     device=device,
        #     num_workers=num_workers,
        #     model_path='/model_rec-46.pt'
        # )

        # outfitGAN
        # compatibility_score, grd_compatibility_score = eval_utils.evaluate_compatibility_given_data(
        #     outfit_dataset,
        #     grd_outfit_dataset,
        #     gen_imgs,
        #     cnn_feat_path,
        #     cnn_feat_gen_path,
        #     args.pretrained_evaluator_ckpt,
        #     batch_size=args.batch_size,
        #     device=device,
        #     num_workers=num_workers
        # )
        # torch.cuda.empty_cache()
        # compatibility_score = compatibility_score.to('cpu').tolist()
        # # grd_compatibility_score = grd_compatibility_score.to('cpu').tolist()
        # 
        # grouped_list = []
        # for i in range(0, len(compatibility_score), args.num_per_outfit):
        #     group = compatibility_score[i:i + args.num_per_outfit]
        #     grouped_list.append(group)

        num_com = 0
        new_com_scores = [com_scores[i:i + args.num_per_outfit] for i in range(0, len(com_scores), args.num_per_outfit)]

        for uid in gen_data:
            for oid in gen_data[uid]:
                gen_data[uid][oid]["eva_compatibility_score"] = new_com_scores[num_com]
                rounded_num = [round(num) for num in new_com_scores[num_com]]
                gen_data[uid][oid]["eva_compatibility_score_round"] = rounded_num
                num_com += 1

        # np.save(eval_save_path, np.array(all_eval_metrics))
        print("Finish Compatibility Score")
    else:
        print("Already evaluate competibility, skip.")
    # -------------------------------------------------------------- #
    #             get each image's personalization score             #
    # -------------------------------------------------------------- #
    if not args.continue_flag:
        gen4personal_sim = {}
        gen4personal_sim["gen"] = []
        gen4personal_sim["hist"] = []
        for uid in gen_data:
            for oid in gen_data[uid]:
                for i, img_path in enumerate(gen_data[uid][oid]["image_paths"]):
                    cate = gen_data[uid][oid]["cates"][0].item()
                    gen4personal_sim["hist"].append(history[uid][cate])

                    if isinstance(img_path, list):
                        gen_img_path = img_path[0]
                        gen_img_path = os.path.join(f"../FashionDPO/", gen_img_path)
                    else:
                        gen_img_path = os.path.join(f"../FashionDPO/", img_path)
                    im = Image.open(gen_img_path)
                    gen4personal_sim["gen"].append(img_trans(im))

        gen_dataset_personal_sim = FashionPersonalSimDataset(gen4personal_sim)
        personal_sim_score = eval_utils.evaluate_personalization_given_data_sim(
            gen4eval=gen_dataset_personal_sim,
            batch_size=args.batch_size,
            device=device,
            num_workers=num_workers,
            similarity_func=args.sim_func
        )
        torch.cuda.empty_cache()

        per_list = []
        for i in range(0, len(personal_sim_score), args.num_per_outfit):
            group = personal_sim_score[i:i + args.num_per_outfit]
            per_list.append(group)

        num_per = 0
        num_per_total = 1000
        for uid in gen_data:
            for oid in gen_data[uid]:
                gen_data[uid][oid]["eva_personal_score"] = per_list[num_per]
                rounded_num = [round(num) for num in per_list[num_per]]
                gen_data[uid][oid]["eva_personal_score_round"] = rounded_num
                num_per += 1
        del gen4personal_sim

        print("Finish Personalization Score")
    else:
        print("Already evaluate personalization, skip.")

    # -------------------------------------------------------------- #
    #                get each image's MiniCPM score                  #
    # -------------------------------------------------------------- #
    model_minicpm = AutoModel.from_pretrained("../FashionDPO/models/huggingface/openbmbMiniCPM-Llama3-V-2_5",
                                      trust_remote_code=True, torch_dtype=torch.float16)
    model_minicpm = model_minicpm.to(device)

    tokenizer = AutoTokenizer.from_pretrained("../FashionDPO/models/huggingface/openbmbMiniCPM-Llama3-V-2_5",
                                              trust_remote_code=True)
    model_minicpm.eval()


    # data prepare
    if not args.continue_flag:
        scores_qua = []
        # scores_com = []
    else:
        scores_qua_np = np.load(eval_save_path_qua, allow_pickle=True)
        # scores_com_np = np.load(eval_save_path_com, allow_pickle=True)
        scores_qua = scores_qua_np.tolist()
        # scores_com = scores_com_np.tolist()
    minicpm_qua = {}
    minicpm_qua["image"] = []
    minicpm_qua["prompt"] = []
    # minicpm_com = {}
    # minicpm_com["image"] = []
    # minicpm_com["prompt"] = []
    for uid in gen_data:
        for oid in gen_data[uid]:
            cate = gen_data[uid][oid]["cates"][0].item()
            cate_res = cate_trans(cate, id_cate_dict)
            if args.evanumber == 5:
                prompt_qua_ori = '''
                This task involves evaluating the generated fashion items. {insert}. 
                Consider whether the fashion elements in the image are complete and whether they conform to fashion design principles. 
                The goal is to classify the quality into one of the following categories:
                1-Poor Quality, 2-Low Quality, 3-Moderate Quality, 4-High Quality, 5-Exceptional Quality. 
                You must reply with one of the categories exactly as listed: 1-Incompatible, 2-Slightly Compatible, 3-Moderately Compatible, 4-Very Compatible, 5-Perfectly Compatible. 
                Even if the image quality is low or if there is minimal information, you must choose the closest possible category. Please provide the best possible category based on the available information.
                Do not provide any other text or explanation. If the response does not follow this format, it will be considered invalid.
                '''
            if args.evanumber == 10:
                prompt_qua_ori = '''
                This task involves evaluating the generated fashion items. {insert}. 
                Consider whether the fashion elements in the image are complete and whether they conform to fashion design principles. 
                The goal is to classify the quality into one of the following categories:
                1-Very Poor Quality, 2-Poor Quality, 3-Low Quality, 4-Below Average Quality, 5-Moderate Quality, 6-Above Average Quality, 7-Good Quality, 8-Very Good Quality, 9-High Quality, 10-Exceptional Quality. 
                You must reply with one of the categories exactly as listed: 1-Very Incompatible, 2-Incompatible, 3-Moderately Compatible, 4-Very Compatible, 5-Perfectly Compatible, 6-Above Average Quality, 7-Good Quality, 8-Very Good Quality, 9-High Quality, 10-Exceptional Quality. 
                Even if the image quality is low or if there is minimal information, you must choose the closest possible category. Please provide the best possible category based on the available information.
                Please choose only one category from the list and provide no other text.
                '''
            prompt_qua = prompt_qua_ori.format(insert=cate_res)

            if args.evanumber == 5:
                prompt_com = '''
                This task involves evaluating the compatibility of a generated outfit based on a set of given fashion items. 
                The model is provided with three items outside the red box and generates a recommended item inside the red box. 
                The goal is to classify the overall outfit compatibility into one of the following categories:1-Incompatible, 2-Slightly Compatible, 3-Moderately Compatible, 4-Very Compatible, 5-Perfectly Compatible. 
                You must reply with one of the categories exactly as listed: 1-Incompatible, 2-Slightly Compatible, 3-Moderately Compatible, 4-Very Compatible, 5-Perfectly Compatible. 
                Even if the image quality is low or if there is minimal information, you must choose the closest possible category. Please provide the best possible category based on the available information.
                Do not provide any other text or explanation. If the response does not follow this format, it will be considered invalid.
                '''
            if args.evanumber == 10:
                prompt_com = '''
                This task involves evaluating the compatibility of a generated outfit based on a set of given fashion items. 
                The model is provided with three items outside the red box and generates a recommended item inside the red box.
                The goal is to classify the overall outfit compatibility into one of the following categories:1-Very Incompatible, 2-Incompatible, 3-Slightly Incompatible, 4-Slightly Compatible, 5-Moderately Compatible, 6-Fairly Compatible, 7-Compatible, 8-Very Compatible, 9-Highly Compatible, 10-Perfectly Compatible. 
                You must reply with one of the categories exactly as listed: 1-Incompatible, 2-Slightly Compatible, 3-Moderately Compatible, 4-Very Compatible, 5-Perfectly Compatible, 6-Fairly Compatible, 7-Compatible, 8-Very Compatible, 9-Highly Compatible, 10-Perfectly Compatible. 
                Even if the image quality is low or if there is minimal information, you must choose the closest possible category. Please provide the best possible category based on the available information.
                Please choose only one category from the list and provide no other text.
                '''


            # for outfit_path in gen_data[uid][oid]["outfit_paths"]:
            #     outfit_path = os.path.join(f"../FashionDPO/", outfit_path)
            #     outfit = Image.open(outfit_path).convert('RGB')
            #     minicpm_com["image"].append(np.array(outfit))
            #     minicpm_com["prompt"].append(prompt_com)
            for img_path in gen_data[uid][oid]["image_paths"]:
                if isinstance(img_path, list):
                    gen_img_path = img_path[0]
                    gen_img_path = os.path.join(f"../FashionDPO/", gen_img_path)
                else:
                    gen_img_path = os.path.join(f"../FashionDPO/", img_path)
                gen_img_path = os.path.join(f"../FashionDPO/", gen_img_path)
                im = Image.open(gen_img_path).convert('RGB')
                minicpm_qua["image"].append(np.array(im))
                minicpm_qua["prompt"].append(prompt_qua)
    minicpm_qua_dataset = FashionCPMquaSimDataset(minicpm_qua)
    # minicpm_com_dataset = FashionCPMcomSimDataset(minicpm_com)



    #----------get MiniCPM quality score----------#
    # scores_qua = eval_utils.evaluate_minicpm_qua(
    #     minicpm_qua_dataset=minicpm_qua_dataset,
    #     batch_size=args.batch_size,
    #     device=device,
    #     num_workers=1
    # )
    # quality_list = []
    # for i in range(0, len(scores_qua), args.num_per_outfit):
    #     quality_group = scores_qua[i:i + args.num_per_outfit]
    #     quality_list.append(quality_group)
    #
    # num_qua = 0
    # for uid in gen_data:
    #     for oid in gen_data[uid]:
    #         gen_data[uid][oid]["eva_minicpm_quality_score"] = quality_list[num_qua]
    #         num_qua += 1
    # print("Finish MiniCPM Quality Score")

    with tqdm(total=len(minicpm_qua["image"]), desc="Get MiniCPM quality score") as pbar:
        for i in range(len(minicpm_qua["image"])):
            if i < len(scores_qua):
                continue
            prompt_img = minicpm_qua["prompt"][i]
            img_gen = Image.fromarray(minicpm_qua["image"][i])
            msgs_qua = [{'role': 'user', 'content': prompt_img}]

            while True:
                res_qua = model_minicpm.chat(
                    image=img_gen,
                    msgs=msgs_qua,
                    tokenizer=tokenizer,
                    sampling=True,
                    temperature=0.1
                )
                if args.evanumber == 5:
                    score_qua = extract_score5(res_qua)
                if args.evanumber == 10:
                    score_qua = extract_score10(res_qua)
                # with open(txt_path, "a") as file:
                #     file.write("Quality " + str(i) + " : " + res_qua + "\n")
                if score_qua != "Score not found":
                    break
                else:
                    print(res_qua)

            scores_qua.append(int(score_qua))
            print(score_qua)
            pbar.update(1)

    quality_list = []
    for i in range(0, len(scores_qua), args.num_per_outfit):
        quality_group = scores_qua[i:i + args.num_per_outfit]
        quality_list.append(quality_group)

    num_qua = 0
    for uid in gen_data:
        for oid in gen_data[uid]:
            gen_data[uid][oid]["eva_minicpm_quality_score"] = quality_list[num_qua]
            num_qua += 1
    print("Finish MiniCPM Quality Score")


    # # ----------get MiniCPM compatibility score----------#
    # with tqdm(total=len(minicpm_com["image"]), desc="Get MiniCPM compatibility score") as pbarcom:
    #     for i in range(len(minicpm_com["image"])):
    #         if i < len(scores_com):
    #             continue
    #         prompt_outfit = minicpm_com["prompt"][i]
    #         outfit_gen = Image.fromarray(minicpm_com["image"][i])
    #         msgs_com = [{'role': 'user', 'content': prompt_outfit}]
    #
    #         while True:
    #             res_com = model_minicpm.chat(
    #                 image=outfit_gen,
    #                 msgs=msgs_com,
    #                 tokenizer=tokenizer,
    #                 sampling=True,
    #                 temperature=0.1
    #             )
    #             if args.evanumber == 5:
    #                 score_com = extract_score5(res_com)
    #             if args.evanumber == 10:
    #                 score_com = extract_score10(res_com)
    #             with open(txt_path, "a") as file:
    #                 file.write("Compatibility " + str(i) + " : " + res_com + "\n")
    #             if score_com != "Score not found":
    #                 break
    #         scores_com.append(int(score_com))
    #         print(score_com)
    #         pbarcom.update(1)
    #
    # cpmcom_list = []
    # for i in range(0, len(scores_com), args.num_per_outfit):
    #     cpmcom_group = scores_com[i:i + args.num_per_outfit]
    #     cpmcom_list.append(cpmcom_group)
    #
    # num_cpmcom = 0
    # for uid in gen_data:
    #     for oid in gen_data[uid]:
    #         gen_data[uid][oid]["eva_minicpm_compatibility_score"] = cpmcom_list[num_cpmcom]
    #         num_cpmcom += 1
    # print("Finish MiniCPM Compatibility Score")


    print(f"Successfully saved evaluation results of {args.eval_version} checkpoint-{ckpt} to {eval_save_path}.")
    np.save(eval_save_path, np.array(gen_data))
    np.save(eval_save_path_qua, np.array(scores_qua))
    # np.save(eval_save_path_com, np.array(scores_com))


    # num_step = 0
    # for uid in gen_data:
    #     for oid in gen_data[uid]:
    #         all_personal_score = []
    #         all_personal_score_int = []
    #         all_minicpm_quality_score = []
    #         all_minicpm_compatibility_score = []
    #
    #         for i, img_path in enumerate(gen_data[uid][oid]["image_paths"]):
    #
    #             cate = gen_data[uid][oid]["cates"][0].item()
    #             personal_sim_score = None
    #
    #             if isinstance(img_path, list):
    #                 gen_img_path = img_path[0]
    #                 gen_img_path = os.path.join(f"../FashionDPO/", gen_img_path)
    #             else:
    #                 gen_img_path = os.path.join(f"../FashionDPO/", img_path)
    #             im = Image.open(gen_img_path)
    #             # gen4personal_sim["hist"].append(history[uid][cate])
    #             # gen4personal_sim["gen"].append(img_trans(im))
    #             gen4personal_hist = history[uid][cate] # Tensor:(1024, )
    #             gen4personal_image = img_trans(im) # Tensor:(3,224,224)
    #
    #             personal_sim_score = eval_utils.evaluate_personalization_for_each_gen(
    #                 gen4eval_hist=gen4personal_hist,
    #                 gen4eval_image=gen4personal_image,
    #                 batch_size=args.batch_size,
    #                 device=device,
    #                 num_workers=num_workers,
    #                 similarity_func=args.sim_func
    #             )
    #             personal_sim_score = personal_sim_score.cpu().detach() # Tensor:(1,)
    #             all_personal_score.append(personal_sim_score.item())
    #             all_personal_score_int.append(round(personal_sim_score.item())) # save integer
    #
    #         gen_data[uid][oid]["personal_score"] = all_personal_score
    #         gen_data[uid][oid]["personal_score_int"] = all_personal_score_int
    #         num_step += 1
    #         print("progress " + str(num_step) + " finish")



def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    main()
