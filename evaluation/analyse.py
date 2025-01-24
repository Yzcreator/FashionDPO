import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter
import argparse
import random

score_name = './final_score_10.jpg'
def parse_all_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument('--num_grade', type=int, default=10)
    parser.add_argument('--save_img', type=str, default=True)
    parser.add_argument('--analyse', type=str, default=True)
    parser.add_argument('--num_per_outfit', type=int, default=7)
    parser.add_argument('--output_dir', type=str, default="../FashionDPO/output/sample_ddim/sample_5_ifashion_7_1000/eval-test-git")
    parser.add_argument('--weight_com', type=float, default=0.3333)
    parser.add_argument('--weight_per', type=float, default=0.3333)
    parser.add_argument('--weight_qua', type=float, default=0.3333)

    args = parser.parse_args()
    return args



def count_scores_per(scores):
    score_counts = Counter(scores)
    score_range = list(range(101))
    counts = [score_counts.get(score, 0) for score in score_range]
    return score_range, counts

def count_scores_com(scores):
    # 统计每个分数出现的次数，保留两位小数
    score_counts = Counter(scores)

    # 创建一个包含0.00到1.00每个分数的列表，步长为0.01
    score_range = list(range(301))

    # 使用列表推导式填充每个分数的出现次数
    counts = [score_counts.get(score, 0) for score in score_range]

    return score_range, counts


def count_scores_normalization(scores):
    # 统计每个分数出现的次数，保留两位小数
    rounded_scores = [round(score, 2) for score in scores]
    score_counts = Counter(rounded_scores)

    # 创建一个包含-1.00到1.00每个分数的列表，步长为0.01
    score_range = [round(x * 0.01 - 1, 2) for x in range(201)]

    # 使用列表推导式填充每个分数的出现次数
    counts = [score_counts.get(score, 0) for score in score_range]

    return score_range, counts

def count_scores_minicpm_qua_5(scores):
    scores = [int(score) if isinstance(score, str) else score for score in scores]
    score_counts = Counter(scores)
    score_range = list(range(1,6))
    counts = [score_counts.get(score, 0) for score in score_range]
    return score_range, counts


def count_scores_minicpm_com_5(scores):
    scores = [int(score) if isinstance(score, str) else score for score in scores]
    score_counts = Counter(scores)
    score_range = list(range(1,6))
    counts = [score_counts.get(score, 0) for score in score_range]
    return score_range, counts


def count_scores_minicpm_qua_10(scores):
    scores = [int(score) if isinstance(score, str) else score for score in scores]
    score_counts = Counter(scores)
    score_range = list(range(1,11))
    counts = [score_counts.get(score, 0) for score in score_range]
    return score_range, counts


def count_scores_minicpm_com_10(scores):
    scores = [int(score) if isinstance(score, str) else score for score in scores]
    score_counts = Counter(scores)
    score_range = list(range(1,11))
    counts = [score_counts.get(score, 0) for score in score_range]
    return score_range, counts

def count_final_scores(scores):
    scores = [int(score) if isinstance(score, str) else score for score in scores]
    score_counts = Counter(scores)
    score_range = list(range(101))
    counts = [score_counts.get(score, 0) for score in score_range]
    return score_range, counts


def plot_histogram_per(score_range, counts, filename='test'):
    plt.figure(figsize=(10, 6))
    plt.bar(score_range, counts, width=1, edgecolor='black')
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.title('Personalization Score Distribution')
    plt.xticks(range(0, 101, 10))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(filename)
    plt.close()

def plot_histogram_per_normalization(score_range, counts, filename='test'):
    plt.figure(figsize=(10, 6))
    plt.bar(score_range, counts, width=0.02, edgecolor='black')
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.title('Personalization Score Distribution')
    ticks = np.arange(0, 1, 0.1)
    plt.xticks(np.arange(0, 1, 0.1))
    plt.gca().set_xticklabels([f'{tick:.2f}' for tick in ticks])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(filename)
    plt.close()

def plot_histogram_com_normalization(score_range, counts, filename='test'):
    plt.figure(figsize=(10, 6))
    plt.bar(score_range, counts, width=0.01, edgecolor='black')
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.title('Compatibility Score Distribution')
    ticks = np.arange(0, 1, 0.1)
    plt.xticks(np.arange(0, 1, 0.1))
    plt.gca().set_xticklabels([f'{tick:.2f}' for tick in ticks])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlim(0, 1)
    plt.savefig(filename)
    plt.close()

def plot_minicpm_qua_normalization(score_range, counts, filename='score_distribution.png'):
    plt.figure(figsize=(10, 6))
    plt.bar(score_range, counts, width=0.1, edgecolor='black')
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
  

    plt.title('MiniCPM Quality Score Distribution')
    ticks = np.arange(0, 1, 0.1)
    plt.xticks(np.arange(0, 1, 0.1))
    plt.gca().set_xticklabels([f'{tick:.2f}' for tick in ticks])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(filename)
    plt.close()

def plot_final_score_normalization(score_range, counts, filename='final_score_distribution.png'):
    plt.figure(figsize=(10, 6))
    plt.bar(score_range, counts, width=0.02, edgecolor='black')
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.title('Final Score Distribution')
    ticks = np.arange(0, 1, 0.1)
    plt.xticks(np.arange(0, 1, 0.1))
    plt.gca().set_xticklabels([f'{tick:.2f}' for tick in ticks])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(filename)
    plt.close()


def plot_histogram_com(score_range, counts, filename='score_distribution.png'):
    plt.figure(figsize=(10, 6))
    plt.bar(score_range, counts, width=1, edgecolor='black')
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.title('Compatibility Score Distribution')
    plt.xticks([round(x) for x in range(301)])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(filename)
    plt.close()

def plot_minicpm_com(score_range, counts, filename='score_distribution.png'):
    plt.figure(figsize=(10, 6))
    plt.bar(score_range, counts, width=0.5, edgecolor='black')
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.title('MiniCPM Compatibility Score Distribution')
    plt.xticks(score_range)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(filename)
    plt.close()

def plot_minicpm_qua(score_range, counts, filename='score_distribution.png'):
    plt.figure(figsize=(10, 6))
    plt.bar(score_range, counts, width=0.5, edgecolor='black')
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.title('MiniCPM Quality Score Distribution')
    plt.xticks(score_range)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(filename)
    plt.close()


def plot_final_score(score_range, counts, filename='final_score_distribution.png'):
    plt.figure(figsize=(10, 6))
    plt.bar(score_range, counts, width=0.5, edgecolor='black')
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.title('Final Score Distribution')
    plt.xticks(range(0, 101, 10))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(filename)
    plt.close()


def main():
    args = parse_all_args()

    npy_file = os.path.join(args.output_dir, f"eval_results_{args.num_grade}.npy")
    eval_result = np.load(npy_file, allow_pickle=True).item()
    eval_save_path = os.path.join(args.output_dir, f"eval_final_results_{args.num_grade}.npy")
    final_score_img_path = f"eval_img/sample_0_ifashion_FITB_{args.num_grade}.jpg"

    # save the score distribution from feedback
    if args.save_img == True:
        personalization_scores = []
        compatibility_scores = []
        minicpm_quality_scores = []
        # minicpm_compatibility_scores = []
        for uid in eval_result:
            for oid in eval_result[uid]:
                # personalization score
                for per_score in eval_result[uid][oid]["eva_personal_score_round"]:
                    personalization_scores.append(per_score)
                # compatibility score
                for com_score in eval_result[uid][oid]["eva_compatibility_score_round"]:
                    compatibility_scores.append(com_score)
                # minicpm_quality score
                for minicpm_quality_score in eval_result[uid][oid]["eva_minicpm_quality_score"]:
                    minicpm_quality_scores.append(minicpm_quality_score)
                # minicpm_compatibility score
                # for minicpm_compatibility_score in eval_result[uid][oid]["eva_minicpm_compatibility_score"]:
                #     minicpm_compatibility_scores.append(minicpm_compatibility_score)

        personalization_score_range, personalization_counts = count_scores_per(personalization_scores)
        plot_histogram_per(personalization_score_range, personalization_counts, './personalization_score.jpg')
        compatibility_score_range, compatibility_counts = count_scores_com(compatibility_scores)
        plot_histogram_com(compatibility_score_range, compatibility_counts,'./compatibility_score.jpg')

        if args.num_grade == 5:
            minicpm_quality_scores_range, minicpm_quality_scores_counts = count_scores_minicpm_qua_5(minicpm_quality_scores)
            plot_minicpm_qua(minicpm_quality_scores_range, minicpm_quality_scores_counts,'./minicpm_quality_score_5.jpg')
            # minicpm_compatibility_scores_range, minicpm_compatibility_scores_counts = count_scores_minicpm_com_5(minicpm_compatibility_scores)
            # plot_minicpm_com(minicpm_compatibility_scores_range, minicpm_compatibility_scores_counts,'./minicpm_compatibility_score_5.jpg')
        if args.num_grade == 10:
            minicpm_quality_scores_range, minicpm_quality_scores_counts = count_scores_minicpm_qua_10(minicpm_quality_scores)
            plot_minicpm_qua(minicpm_quality_scores_range, minicpm_quality_scores_counts, './minicpm_quality_score_10.jpg')
            # minicpm_compatibility_scores_range, minicpm_compatibility_scores_counts = count_scores_minicpm_com_10(minicpm_compatibility_scores)
            # plot_minicpm_com(minicpm_compatibility_scores_range, minicpm_compatibility_scores_counts,'./minicpm_compatibility_score_10.jpg')

    # analyse the feedback score, mix_max_normalization, weighted summation, select according to the threshold
    if args.analyse == True:
        all_final_scores = []
        all_final_scores_round = []
        max_in_group = []
        min_in_group = []
        personalization_score_np = np.array([])
        compatibility_score_np = np.array([])
        minicpm_quality_score_np = np.array([])
        for uid in eval_result:
            for oid in eval_result[uid]:
                personalization_score_np = np.append(personalization_score_np, eval_result[uid][oid]["eva_personal_score_round"])
                compatibility_score_np = np.append(compatibility_score_np, eval_result[uid][oid]["eva_compatibility_score_round"])
                minicpm_quality_score_np = np.append(minicpm_quality_score_np, eval_result[uid][oid]["eva_minicpm_quality_score"])

        P = personalization_score_np
        P_min = np.min(personalization_score_np)
        P_max = np.max(personalization_score_np)
        P_norm = (P - P_min) / (P_max - P_min)
        # P_median_norm = np.median(P_norm)
        # P_shifted = P_norm - P_median_norm + 0.5
        # P_shifted = P_shifted.tolist()
        P_shifted = P_norm.tolist()

        C = compatibility_score_np
        C_min = np.min(compatibility_score_np)
        C_max = np.max(compatibility_score_np)
        C_norm = (C - C_min) / (C_max - C_min)
        # C_median_norm = np.median(C_norm)
        # C_shifted = C_norm - C_median_norm + 0.5
        # C_shifted = C_shifted.tolist()
        C_shifted = C_norm.tolist()

        MQ = minicpm_quality_score_np
        MQ_min = np.min(minicpm_quality_score_np)
        MQ_max = np.max(minicpm_quality_score_np)
        MQ_norm = (MQ - MQ_min) / (MQ_max - MQ_min)
        # MQ_median_norm = np.median(MQ_norm)
        # MQ_shifted = MQ_norm - MQ_median_norm + 0.5
        # MQ_shifted = MQ_shifted.tolist()
        MQ_shifted = MQ_norm.tolist()

        for i in range(len(P_shifted)):
            final_score = args.weight_per * P_shifted[i] + args.weight_com * C_shifted[i] + args.weight_qua * MQ_shifted[i]
            all_final_scores.append(final_score)
            all_final_scores_round.append(round(final_score,2))

        sub_all_final_scores = [all_final_scores[i:i+args.num_per_outfit] for i in range(0, len(all_final_scores), args.num_per_outfit)]
        sub_all_final_scores_round = [all_final_scores_round[i:i + args.num_per_outfit] for i in
                                range(0, len(all_final_scores_round), args.num_per_outfit)]
        num = 0
        for uid in eval_result:
            for oid in eval_result[uid]:
                eval_result[uid][oid]["final_score"] = sub_all_final_scores[num]
                eval_result[uid][oid]["final_score_round"] = sub_all_final_scores_round[num]
                num += 1

        if args.save_img == True:
            P_range, P_counts = count_scores_normalization(P_shifted)
            plot_histogram_per_normalization(P_range, P_counts, './P_score.jpg')
            C_range, C_counts = count_scores_normalization(C_shifted)
            plot_histogram_com_normalization(C_range, C_counts, './C_score.jpg')
            if args.num_grade == 10:
                MQ_range, MQ_counts = count_scores_normalization(MQ_shifted)
                plot_minicpm_qua_normalization(MQ_range, MQ_counts,'./MQ_score_10.jpg')


        # Using median_score as threshold, then randomly select one from the positive images and one from the negative images to form a pair
        threshold = np.median(sub_all_final_scores)
        pos = 0
        neg = 0
        for uid in eval_result:
            for oid in eval_result[uid]:
                positive_score_list = []
                negative_score_list = []
                for i, score in enumerate(eval_result[uid][oid]["final_score"]):
                    if score >= threshold:
                        positive_score_list.append(i)
                        pos += 1
                    else:
                        negative_score_list.append(i)
                        neg += 1
                if not positive_score_list:
                    print("Attention! Positive_score_list is empty")
                    print(eval_result[uid][oid]["final_score"])
                if not negative_score_list:
                    print("Attention! Negative_score_list is empty")
                    print(eval_result[uid][oid]["final_score"])
                # positive_result = random.choice(positive_result)
                # negative_result = random.choice(negative_result)
                eval_result[uid][oid]["feedback_results"] = []
                for i in range(args.num_per_outfit):
                    if i in positive_score_list:
                        eval_result[uid][oid]["feedback_results"].append(1)
                    else:
                        eval_result[uid][oid]["feedback_results"].append(0)

        np.save(eval_save_path, eval_result)

        # draw final score plot
        if args.save_img == True:
            final_score_range, final_counts = count_scores_normalization(all_final_scores)
            plot_final_score_normalization(final_score_range, final_counts, score_name)





if __name__ == "__main__":
    main()
