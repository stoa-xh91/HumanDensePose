import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy

COCO_PERSON_KEYPOINT_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle","right_ankle",
]

COCO_PERSON_SURFACE_NAMES = [
    "back_torso", "front_torso",
    "right_hand", "left_hand",
    "left_foot", "right_foot",
    "right_leg", "left_leg",
    "right_leg", "left_leg",
    "right_leg", "left_leg",
    "right_leg", "left_leg",
    "left_arm", "right_arm",
    "left_arm","right_arm",
    "left_arm","right_arm",
    "left_arm","right_arm",
    "left_head","right_head"
]
COCO_PERSON_PART_NAMES = [
    "torso",
    "right_hand", "left_hand",
    "left_foot", "right_foot",
    "upper_right_leg", "upper_left_leg",
    "lower_right_leg", "lower_left_leg",
    "upper_left_arm", "upper_right_arm",
    "lower_left_arm","lower_right_arm",
    "head"
]
part_to_surface_rules = {1: [1, 2],
                      2: [3],
                      3: [4],
                      4: [5],
                      5: [6],
                      6: [7, 9],
                      7: [8, 10],
                      8: [11, 13],
                      9: [12, 14],
                      10: [15, 17],
                      11: [16, 18],
                      12: [19, 21],
                      13: [20, 22],
                      14: [23, 24]}

kp_to_surface_rules = {1: [23, 24],
                      2: [23, 24],
                      3: [23, 24],
                      4: [23, 24],
                      5: [23, 24],
                      6: [1, 2, 15, 17],
                      7: [1, 2, 16, 18],
                      8: [15, 17, 19, 21],
                      9: [16, 18, 20, 22],
                      10: [4, 19, 21],
                      11: [3, 20, 22],
                      12: [1, 2, 8, 10],
                      13: [1, 2, 7, 9],
                      14: [8, 10, 12, 14],
                      15: [7, 9, 11, 13],
                      16: [5, 12, 14],
                      17: [6, 11, 13]}

kpt_labels = ["nose", "eye",
                "ear", "shoulder",
                "elbow", "wrist",
                "hip", "knee","ankle"]

part_labels = ["torso", "hand",
                "foot", "leg",
                "arm", "head"]

bbox_labels = ["person"]

kpt_nodes_2d = set(kpt_labels)

part_nodes_2d = set(part_labels)

bbox_nodes_2d = set(bbox_labels)

vg_relation_gt = json.load(open('relationships.json'))

surface_labels = ["torso", "hand","foot", "leg","arm", "head"]

def get_kpt_surf_rel_matrix():
    rels = {}
    for p in kpt_labels:
        rels[p] = {}
        for sub_p in surface_labels:
            rels[p][sub_p] = 0
    return  rels

def get_full_kpt_surf_matrix():
    rels = {}
    for p in COCO_PERSON_KEYPOINT_NAMES:
        rels[p] = {}
        for sub_p in COCO_PERSON_SURFACE_NAMES:
            rels[p][sub_p] = 0
    return  rels

def get_part_surf_rel_matrix():
    rels = {}
    for p in part_labels:
        rels[p] = {}
        for sub_p in surface_labels:
            rels[p][sub_p] = 0
    return  rels

def get_full_part_surf_matrix():
    rels = {}
    for p in COCO_PERSON_PART_NAMES:
        rels[p] = {}
        for sub_p in COCO_PERSON_SURFACE_NAMES:
            rels[p][sub_p] = 0
    return  rels

def get_bbox_surf_rel_matrix():
    rels = {}
    for p in bbox_labels:
        rels[p] = {}
        for sub_p in surface_labels:
            rels[p][sub_p] = 0
    return  rels

def col_row_normalize(matrix):
    row_matrix = np.sum(matrix,axis=1).reshape((matrix.shape[0],1))
    col_matrix = np.sum(matrix,axis=0).reshape((1,matrix.shape[1]))
    norms = np.sqrt(np.matmul(row_matrix, col_matrix))
    normed_matrix = matrix / norms
    return normed_matrix

def col_row_normalize_with_id(matrix):
    row_matrix = np.sum(matrix,axis=1).reshape((matrix.shape[0],1))
    col_matrix = np.sum(matrix,axis=0).reshape((1,matrix.shape[1]))
    norms = np.sqrt(np.matmul(row_matrix, col_matrix))
    normed_matrix = matrix / norms
    return normed_matrix

def expand_to_complete_kpt_surf_relations(rel_maps):
    kpt_rel_matrix = np.zeros((len(COCO_PERSON_KEYPOINT_NAMES), len(COCO_PERSON_SURFACE_NAMES)))
    kpt_maps = get_full_kpt_surf_matrix()
    for i in range(len(kpt_labels)):
        if "nose" in kpt_labels[i]:
            l_sub = "nose"
            r_sub = "nose"
        else:
            l_sub = "left_"+kpt_labels[i]
            r_sub = "right_"+kpt_labels[i]

        for j in range(len(surface_labels)):
            if "torso" in surface_labels[j]:
                l_obj = "back_" + surface_labels[j]
                r_obj = "front_" + surface_labels[j]
            else:
                l_obj = "left_" + surface_labels[j]
                r_obj = "right_" + surface_labels[j]

            kpt_maps[l_sub][l_obj] = rel_maps[kpt_labels[i]][surface_labels[j]]
            kpt_maps[l_sub][r_obj] = rel_maps[kpt_labels[i]][surface_labels[j]]
            kpt_maps[r_sub][l_obj] = rel_maps[kpt_labels[i]][surface_labels[j]]
            kpt_maps[r_sub][r_obj] = rel_maps[kpt_labels[i]][surface_labels[j]]
    for i in range(len(COCO_PERSON_KEYPOINT_NAMES)):
        for j in range(len(COCO_PERSON_SURFACE_NAMES)):
            kpt_rel_matrix[i,j] = kpt_maps[COCO_PERSON_KEYPOINT_NAMES[i]][COCO_PERSON_SURFACE_NAMES[j]]
    
    kpt_rel_matrix = col_row_normalize(kpt_rel_matrix)
    common_relations = np.zeros((17, 24), dtype=np.float32)
    for k in kp_to_surface_rules.keys():
        for v in kp_to_surface_rules[k]:
            common_relations[k-1, v-1] = 1
    sem_relations = generate_kp_dp_relations_from_word_vec()
    final_relation = (kpt_rel_matrix + sem_relations + common_relations) * 0.3
    final_relation = np.hstack([np.zeros((final_relation.shape[0], 1)), final_relation])
    return final_relation

def get_statics_kpt_surf_rel_from_vg(relations_maps=None):

    words_related_to_kpt_nodes = [[], [],
                [], ["torso","body"],
                ["arm"],
                ["hand","forearm"],
                ["buttocks","arse","ass","thigh"],
                # ["lap","calf","crus","shank","shin"],
                ["lap","crus","shank","shin"],
                ["ankle","foot"]]
    related_words_to_kpt_words = {}
    for i in range(len(kpt_labels)):
        if len(words_related_to_kpt_nodes[i])>0:
            for w in words_related_to_kpt_nodes[i]:
                related_words_to_kpt_words[w] = kpt_labels[i]

    surface_nodes_3d = set(surface_labels)

    words_related_to_surf_nodes = [["body", "shoulder","hip"], ["glove","gesture"],
                    ["ankle"], [],#["calf","crus","shank","shin"],
                    ["elbow"],["brain","neck"]]
    related_words_to_surface_words = {}
    for i in range(len(surface_labels)):
        if len(words_related_to_surf_nodes[i])>0:
            for w in words_related_to_surf_nodes[i]:
                related_words_to_surface_words[w] = surface_labels[i]
    print("start recording...")
    for item in vg_relation_gt:
        relations = item['relationships']
        for rel in relations:
            sub = None
            obj = None
            if 'name' in rel['subject']:
                sub = rel['subject']['name']
            elif 'names' in rel['subject']:
                sub = rel['subject']['names'][0]
                if len(rel['subject']['names'])>1:
                    print('sub names:',rel['subject']['names'])
            if 'name' in rel['object']:
                obj = rel['object']['name']
            elif 'names' in rel['object']:
                obj = rel['object']['names'][0]
                if len(rel['object']['names'])>1:
                    print('obj names:',rel['object']['names'])
            k_words = set([sub, obj])
            if len(k_words & kpt_nodes_2d)>0 and len(k_words & surface_nodes_3d)>0:
                inter_sub = list(k_words & kpt_nodes_2d)
                inter_obj = list(k_words & surface_nodes_3d)
                for k_sub in inter_sub:
                    for k_obj in inter_obj:
                        relations_maps[k_sub][k_obj] += 1
                continue
            
            if len(k_words & set(related_words_to_kpt_words.keys()))>0 and len(k_words & surface_nodes_3d)>0:
                inter_sub = list(k_words & set(related_words_to_kpt_words.keys()))
                inter_obj = list(k_words & surface_nodes_3d)
                for k_sub in inter_sub:
                    for k_obj in inter_obj:
                        relations_maps[related_words_to_kpt_words[k_sub]][k_obj] += 1
                continue

            if len(k_words & kpt_nodes_2d)>0 and len(k_words & set(related_words_to_surface_words.keys()))>0:
                inter_sub = list(k_words & kpt_nodes_2d)
                inter_obj = list(k_words & set(related_words_to_surface_words.keys()))
                for k_sub in inter_sub:
                    for k_obj in inter_obj:
                        relations_maps[k_sub][related_words_to_surface_words[k_obj]] += 1
                continue

    for k in relations_maps.keys():
        print(k)
        for sub_k in relations_maps[k]:
            print(sub_k, ":", relations_maps[k][sub_k])
    print('extend the matrix to the final one')
    kpt_surf_rel_matrix = expand_to_complete_kpt_surf_relations(relations_maps)
    generate_visulize_results(kpt_surf_rel_matrix, ["bg"] + copy.deepcopy(COCO_PERSON_SURFACE_NAMES), COCO_PERSON_KEYPOINT_NAMES, "kpt_to_surface_")
    return kpt_surf_rel_matrix

def expand_to_complete_part_surf_relations(rel_maps):
    rel_matrix = np.zeros((len(COCO_PERSON_PART_NAMES), len(COCO_PERSON_SURFACE_NAMES)))
    part_maps = get_full_part_surf_matrix()
    for i in range(len(part_labels)):
        if "torso" in part_labels[i] or "head" in part_labels[i]:
            upl_sub = part_labels[i]
            lol_sub = part_labels[i]
            upr_sub = part_labels[i]
            lor_sub = part_labels[i]
        else:
            upl_sub = "left_" + part_labels[i]
            lol_sub = "left_" + part_labels[i]
            upr_sub = "right_" + part_labels[i]
            lor_sub = "right_" + part_labels[i]
        
        if "leg" in part_labels[i] or "arm" in part_labels[i]:
            upl_sub = "upper_" + upl_sub
            lol_sub = "lower_" + lol_sub
            upr_sub = "upper_" + upr_sub
            lor_sub = "lower_" + lor_sub

        for j in range(len(surface_labels)):
            if "torso" in surface_labels[j]:
                l_obj = "back_" + surface_labels[j]
                r_obj = "front_" + surface_labels[j]
            else:
                l_obj = "left_" + surface_labels[j]
                r_obj = "right_" + surface_labels[j]
            
            part_maps[upl_sub][l_obj] = rel_maps[part_labels[i]][surface_labels[j]]
            part_maps[lol_sub][r_obj] = rel_maps[part_labels[i]][surface_labels[j]]
            part_maps[upr_sub][l_obj] = rel_maps[part_labels[i]][surface_labels[j]]
            part_maps[lor_sub][r_obj] = rel_maps[part_labels[i]][surface_labels[j]]
            
    for i in range(len(COCO_PERSON_PART_NAMES)):
        for j in range(len(COCO_PERSON_SURFACE_NAMES)):
            rel_matrix[i,j] = part_maps[COCO_PERSON_PART_NAMES[i]][COCO_PERSON_SURFACE_NAMES[j]]
    
    rel_matrix = col_row_normalize(rel_matrix)
    common_relations = np.zeros((14, 24), dtype=np.float32)
    for k in part_to_surface_rules.keys():
        for v in part_to_surface_rules[k]:
            common_relations[k-1, v-1] = 1
    sem_relations = generate_part_surf_relations_from_word_vec()
    final_relation = (rel_matrix + sem_relations + common_relations) * 0.3
    final_relation = np.hstack([np.zeros((final_relation.shape[0], 1)), final_relation])
    final_relation = np.vstack([np.zeros((1, final_relation.shape[1])), final_relation])
    return final_relation

def get_statics_part_surf_rel_from_vg(relations_maps=None):

    words_related_to_part_nodes = [["body", "shoulder","hip"], ["glove","gesture"],
                    ["ankle"], [],
                    ["elbow"],["brain","neck"]]
    related_words_to_part_words = {}
    for i in range(len(part_labels)):
        if len(words_related_to_part_nodes[i])>0:
            for w in words_related_to_part_nodes[i]:
                related_words_to_part_words[w] = part_labels[i]

    surface_nodes_3d = set(surface_labels)

    words_related_to_surf_nodes = [["body","shoulder","hip"], ["glove","gesture"],
                    ["ankle"], [],
                    ["elbow"],["brain","neck"]]
    related_words_to_surface_words = {}
    for i in range(len(surface_labels)):
        if len(words_related_to_surf_nodes[i])>0:
            for w in words_related_to_surf_nodes[i]:
                related_words_to_surface_words[w] = surface_labels[i]
    print("start recording...")

    for item in vg_relation_gt:
        relations = item['relationships']
        for rel in relations:
            sub = None
            obj = None
            if 'name' in rel['subject']:
                sub = rel['subject']['name']
            elif 'names' in rel['subject']:
                sub = rel['subject']['names'][0]
                if len(rel['subject']['names'])>1:
                    print('sub names:',rel['subject']['names'])
            if 'name' in rel['object']:
                obj = rel['object']['name']
            elif 'names' in rel['object']:
                obj = rel['object']['names'][0]
                if len(rel['object']['names'])>1:
                    print('obj names:',rel['object']['names'])
            k_words = set([sub, obj])
            if len(k_words & part_nodes_2d)>0 and len(k_words & surface_nodes_3d)>0:
                inter_sub = list(k_words & part_nodes_2d)
                inter_obj = list(k_words & surface_nodes_3d)
                for k_sub in inter_sub:
                    for k_obj in inter_obj:
                        relations_maps[k_sub][k_obj] += 1
                continue
            
            if len(k_words & set(related_words_to_part_words.keys()))>0 and len(k_words & surface_nodes_3d)>0:
                inter_sub = list(k_words & set(related_words_to_part_words.keys()))
                inter_obj = list(k_words & surface_nodes_3d)
                for k_sub in inter_sub:
                    for k_obj in inter_obj:
                        relations_maps[related_words_to_part_words[k_sub]][k_obj] += 1
                continue

            if len(k_words & part_nodes_2d)>0 and len(k_words & set(related_words_to_surface_words.keys()))>0:
                inter_sub = list(k_words & part_nodes_2d)
                inter_obj = list(k_words & set(related_words_to_surface_words.keys()))
                for k_sub in inter_sub:
                    for k_obj in inter_obj:
                        relations_maps[k_sub][related_words_to_surface_words[k_obj]] += 1
                continue
    for k in relations_maps.keys():
        print(k)
        for sub_k in relations_maps[k]:
            print(sub_k, ":", relations_maps[k][sub_k])            
    print('extend the matrix to the final one')
    part_surf_rel_matrix = expand_to_complete_part_surf_relations(relations_maps)
    print(part_surf_rel_matrix.shape)
    generate_visulize_results(part_surf_rel_matrix, ["bg"] + copy.deepcopy(COCO_PERSON_SURFACE_NAMES), ["bg"] + COCO_PERSON_PART_NAMES, "part_to_surface_")
    return part_surf_rel_matrix

def expand_to_complete_bbox_surf_relations(rel_maps):
    rel_matrix = np.zeros((len(bbox_labels), len(COCO_PERSON_SURFACE_NAMES)))
    bbox_maps = get_bbox_surf_rel_matrix()
    for i in range(len(bbox_labels)):
        sub = bbox_labels[i]

        for j in range(len(surface_labels)):
            if "torso" in surface_labels[j]:
                l_obj = "back_" + surface_labels[j]
                r_obj = "front_" + surface_labels[j]
            else:
                l_obj = "left_" + surface_labels[j]
                r_obj = "right_" + surface_labels[j]
            
            bbox_maps[sub][l_obj] = rel_maps[bbox_labels[i]][surface_labels[j]]
            bbox_maps[sub][r_obj] = rel_maps[bbox_labels[i]][surface_labels[j]]

            
    for i in range(len(bbox_labels)):
        for j in range(len(COCO_PERSON_SURFACE_NAMES)):
            rel_matrix[i,j] = bbox_maps[bbox_labels[i]][COCO_PERSON_SURFACE_NAMES[j]]
    
    rel_matrix = col_row_normalize(rel_matrix)
    
    sem_relations = generate_bbox_surf_relations_from_word_vec()
    final_relation = (rel_matrix + sem_relations) * 0.5
    final_relation = np.hstack([np.zeros((final_relation.shape[0], 1)), final_relation])
    return final_relation

def get_statics_bbox_surf_rel_from_vg(relations_maps=None):

    words_related_to_bbox_nodes = [["body", "human"]]
    related_words_to_bbox_words = {}
    for i in range(len(bbox_labels)):
        if len(words_related_to_bbox_nodes[i])>0:
            for w in words_related_to_bbox_nodes[i]:
                related_words_to_bbox_words[w] = bbox_labels[i]

    surface_nodes_3d = set(surface_labels)

    words_related_to_surf_nodes = [["body","shoulder","hip"], ["glove","gesture"],
                    ["ankle"], [],
                    ["elbow"],["brain","neck"]]
    related_words_to_surface_words = {}
    for i in range(len(surface_labels)):
        if len(words_related_to_surf_nodes[i])>0:
            for w in words_related_to_surf_nodes[i]:
                related_words_to_surface_words[w] = surface_labels[i]
    print("start recording...")

    for item in vg_relation_gt:
        relations = item['relationships']
        for rel in relations:
            sub = None
            obj = None
            if 'name' in rel['subject']:
                sub = rel['subject']['name']
            elif 'names' in rel['subject']:
                sub = rel['subject']['names'][0]
                if len(rel['subject']['names'])>1:
                    print('sub names:',rel['subject']['names'])
            if 'name' in rel['object']:
                obj = rel['object']['name']
            elif 'names' in rel['object']:
                obj = rel['object']['names'][0]
                if len(rel['object']['names'])>1:
                    print('obj names:',rel['object']['names'])
            k_words = set([sub, obj])
            if len(k_words & bbox_nodes_2d)>0 and len(k_words & surface_nodes_3d)>0:
                inter_sub = list(k_words & bbox_nodes_2d)
                inter_obj = list(k_words & surface_nodes_3d)
                for k_sub in inter_sub:
                    for k_obj in inter_obj:
                        relations_maps[k_sub][k_obj] += 1
                continue
            
            if len(k_words & set(related_words_to_bbox_words.keys()))>0 and len(k_words & surface_nodes_3d)>0:
                inter_sub = list(k_words & set(related_words_to_bbox_words.keys()))
                inter_obj = list(k_words & surface_nodes_3d)
                for k_sub in inter_sub:
                    for k_obj in inter_obj:
                        relations_maps[related_words_to_bbox_words[k_sub]][k_obj] += 1
                continue

            if len(k_words & bbox_nodes_2d)>0 and len(k_words & set(related_words_to_surface_words.keys()))>0:
                inter_sub = list(k_words & bbox_nodes_2d)
                inter_obj = list(k_words & set(related_words_to_surface_words.keys()))
                for k_sub in inter_sub:
                    for k_obj in inter_obj:
                        relations_maps[k_sub][related_words_to_surface_words[k_obj]] += 1
                continue
    for k in relations_maps.keys():
        print(k)
        for sub_k in relations_maps[k]:
            print(sub_k, ":", relations_maps[k][sub_k])            
    print('extend the matrix to the final one')
    rel_matrix = expand_to_complete_bbox_surf_relations(relations_maps)
    rel_matrix = np.repeat(rel_matrix, 6, 0)
    print(rel_matrix)
    generate_visulize_results(rel_matrix, ["bg"] + copy.deepcopy(COCO_PERSON_SURFACE_NAMES), ["person"]*6, "person_to_surface_")
    return rel_matrix

def generate_visulize_results(matrix,labels_x,labels_y, prefix=""):
    ax = sns.heatmap(matrix, cmap=plt.cm.hot)
    plt.title('Confusion_Matrix')
    plt.yticks(range(len(labels_y)), rotation=50, labels=labels_y)
    plt.xticks(range(len(labels_x)), rotation=50, labels=labels_x)
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    plt.savefig(prefix+"comm_relations.png")
    plt.close()


def generate_kp_dp_relations_from_word_vec():
    f = open("glove.6B.300d.txt")
    we = {}
    k_words = kpt_labels + ["middle", "left", "right","back","front"]+ surface_labels
    k_words = set(k_words)
    for str in f.readlines():

        feats = str.split(" ")
        if feats[0] not in k_words:
            continue
        feat_values = []
        for i in range(1,len(feats)):
            feat_values.append(float(feats[i]))
        we[feats[0]] = np.asarray(feat_values)
    kpt_word_emb = np.zeros((len(COCO_PERSON_KEYPOINT_NAMES),600))
    for id, kp in enumerate(COCO_PERSON_KEYPOINT_NAMES):
        if kp == "nose":
            kp = "middle_" + kp
        kp = kp.split("_")
        kpt_word_emb[id,:300] = we[kp[0]]
        kpt_word_emb[id,300:] = we[kp[1]]
    surface_word_emb = np.zeros((len(COCO_PERSON_SURFACE_NAMES), 600))
    for id, surface in enumerate(COCO_PERSON_SURFACE_NAMES):
        surface = surface.split("_")
        surface_word_emb[id, :300] = we[surface[0]]
        surface_word_emb[id, 300:] = we[surface[1]]
    normed_kpt_emb = np.sqrt(np.sum(kpt_word_emb**2, axis=1)).reshape((-1,1))
    normed_sur_emb = np.sqrt(np.sum(surface_word_emb**2, axis=1)).reshape((1,-1))
    norm = np.matmul(normed_kpt_emb, normed_sur_emb)
    sim_matrix = np.matmul(kpt_word_emb, surface_word_emb.transpose())
    sem_relations = sim_matrix / norm
    return sem_relations

def generate_part_surf_relations_from_word_vec():
    f = open("glove.6B.300d.txt")
    we = {}
    k_words = part_labels + ["full","upper","lower","middle", "left", "right","back","front"]+ surface_labels
    k_words = set(k_words)
    for str in f.readlines():

        feats = str.split(" ")
        if feats[0] not in k_words:
            continue
        feat_values = []
        for i in range(1,len(feats)):
            feat_values.append(float(feats[i]))
        we[feats[0]] = np.asarray(feat_values)
    part_word_emb = np.zeros((len(COCO_PERSON_PART_NAMES),900))
    for id, kw in enumerate(COCO_PERSON_PART_NAMES):
        
        if kw == "torso" or kw == "head":
            kw = "full_middle_" + kw
        split_words = kw.split("_")
        if len(split_words)==1:
            kw = "full_middle_" + kw
        elif len(split_words)==2:
            kw = "full_" + kw
        kw = kw.split("_")   
        part_word_emb[id,:300] = we[kw[0]]
        part_word_emb[id,300:600] = we[kw[1]]
        part_word_emb[id,600:900] = we[kw[2]]

    surface_word_emb = np.zeros((len(COCO_PERSON_SURFACE_NAMES), 900))
    for id, surface in enumerate(COCO_PERSON_SURFACE_NAMES):
        surface = "full_"+surface
        surface = surface.split("_")
        surface_word_emb[id, :300] = we[surface[0]]
        surface_word_emb[id, 300:600] = we[surface[1]]
        surface_word_emb[id, 600:900] = we[surface[2]]
    normed_kpt_emb = np.sqrt(np.sum(part_word_emb**2, axis=1)).reshape((-1,1))
    normed_sur_emb = np.sqrt(np.sum(surface_word_emb**2, axis=1)).reshape((1,-1))
    norm = np.matmul(normed_kpt_emb, normed_sur_emb)
    sim_matrix = np.matmul(part_word_emb, surface_word_emb.transpose())
    sem_relations = sim_matrix / norm
    return sem_relations


def generate_bbox_surf_relations_from_word_vec():
    f = open("glove.6B.300d.txt")
    we = {}
    k_words = bbox_labels + ["full","upper","lower","middle", "left", "right","back","front"]+ surface_labels
    k_words = set(k_words)
    for str in f.readlines():

        feats = str.split(" ")
        if feats[0] not in k_words:
            continue
        feat_values = []
        for i in range(1,len(feats)):
            feat_values.append(float(feats[i]))
        we[feats[0]] = np.asarray(feat_values)
    bbox_word_emb = np.zeros((len(bbox_labels),900))
    for id, kw in enumerate(bbox_labels):
        
        kw = "full_middle_" + kw
        kw = kw.split("_")   
        bbox_word_emb[id,:300] = we[kw[0]]
        bbox_word_emb[id,300:600] = we[kw[1]]
        bbox_word_emb[id,600:900] = we[kw[2]]

    surface_word_emb = np.zeros((len(COCO_PERSON_SURFACE_NAMES), 900))
    for id, surface in enumerate(COCO_PERSON_SURFACE_NAMES):
        surface = "full_"+surface
        surface = surface.split("_")
        surface_word_emb[id, :300] = we[surface[0]]
        surface_word_emb[id, 300:600] = we[surface[1]]
        surface_word_emb[id, 600:900] = we[surface[2]]
    normed_kpt_emb = np.sqrt(np.sum(bbox_word_emb**2, axis=1)).reshape((-1,1))
    normed_sur_emb = np.sqrt(np.sum(surface_word_emb**2, axis=1)).reshape((1,-1))
    norm = np.matmul(normed_kpt_emb, normed_sur_emb)
    sim_matrix = np.matmul(bbox_word_emb, surface_word_emb.transpose())
    sem_relations = sim_matrix / norm
    return sem_relations


if __name__ == '__main__':
    
    relations = get_kpt_surf_rel_matrix()
    relations = get_statics_kpt_surf_rel_from_vg(relations)
    pickle.dump(relations, open('kpt_surf_crkg.pkl','wb'))
    relations = get_part_surf_rel_matrix()
    relations = get_statics_part_surf_rel_from_vg(relations)
    pickle.dump(relations, open('part_surf_crkg.pkl','wb'))
    relations = get_bbox_surf_rel_matrix()
    relations = get_statics_bbox_surf_rel_from_vg(relations)
    pickle.dump(relations, open('person_surf_crkg.pkl','wb'))
