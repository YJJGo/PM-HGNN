import argparse
import gc
import logging
import os
import sys
import time
from collections import defaultdict
import torch
import torch.nn as nn
import numpy as np
from ruamel.yaml import YAML
from process import load_data
from model import EMHGNN
from torch_sparse import SparseTensor
from utils import (set_random_seed, get_n_params, EarlyStopping, evaluate, get_feature_sim, get_full_paths,
                   calculate_overlap_ratio, merge_sparse, train, get_jaccard_sim, prune, SparseCalcUtil, get_path_adj)
import warnings
warnings.simplefilter("ignore")
times_list = []


def main(args):
    set_random_seed(args.seed)
    device = 'cuda:{}'.format(args.gpu) if not args.cpu else 'cpu'
    args.device = device
    calc = SparseCalcUtil(device)
    print(f"Load data...")
    start = time.time()
    feats_dict, adjs, idx_shift_dict, labels, num_classes, tgt_type, train_val_test_idx, dl = load_data(args)
    end = time.time()
    print(f'Time used for load data: {end - start}')

    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']
    train_node_nums, val_node_nums, test_node_nums = len(train_idx), len(val_idx), len(test_idx)
    train_val_point, val_test_point = train_node_nums, train_node_nums + val_node_nums
    labeled_idx = np.concatenate((train_idx, val_idx, test_idx))
    labeled_num_nodes = len(labeled_idx)
    print(f'#Train {train_node_nums}, #Val {val_node_nums}, #Test {test_node_nums}')

    num_nodes = dl.nodes['count'][0]
    if labeled_num_nodes < num_nodes:
        flag = np.ones(num_nodes, dtype=bool)
        flag[train_idx] = 0
        flag[val_idx] = 0
        flag[test_idx] = 0
        extra_idx = np.where(flag)[0]
        print(f'Find {len(extra_idx)} extra nid for dataset {args.dataset}')
    else:
        extra_idx = np.array([])

    start = time.time()
    paths = get_full_paths(list(adjs.keys()), tgt_type, args.num_hops)

    hop_adjs = {}
    for path in paths:
        adj = get_path_adj(adjs, hop_adjs, path, calc)
        hop_adjs[path] = adj
    end = time.time()
    print(f'Time used for adjs propagation: {end - start}')

    if args.redundancy_module:
        start = time.time()
        limited_ratio = [float(i) for i in args.ratio.split(',')]
        adjs_path = f'adjs/{args.dataset}/'
        file_name = f'hop_adjs_{args.num_hops}_{limited_ratio[0]}_{limited_ratio[1]}.pt'
        if not os.path.exists(os.path.join(adjs_path, file_name)):
            paths_dict = defaultdict(list)
            for path in paths:
                paths_dict[path[-1]].append(path)
            paths_list = list(paths_dict.values())
            for k in range(len(paths_list)):
                for i in range(len(paths_list[k]) - 1, -1, -1):
                    for j in range(0, i):
                        path_main, path_sub = paths_list[k][j], paths_list[k][i]
                        ratio = calculate_overlap_ratio(hop_adjs[path_sub], hop_adjs[path_main], calc, 'max')
                        if limited_ratio[0] < ratio <= limited_ratio[1]:
                            hop_adjs[path_main] = merge_sparse(hop_adjs[path_sub], hop_adjs[path_main], calc, 'sum')
                            print(f'({path_main}, {path_sub}): {ratio:.4f} sum merge {path_sub}=>{path_main}')
                            del hop_adjs[path_sub]
                            break
                        elif ratio > limited_ratio[1]:
                            hop_adjs[path_main] = merge_sparse(hop_adjs[path_sub], hop_adjs[path_main], calc, 'max')
                            print(f'({path_main}, {path_sub}): {ratio:.4f} max merge {path_sub}=>{path_main}')
                            del hop_adjs[path_sub]
                            break
            # os.makedirs(adjs_path, exist_ok=True)
            # torch.save(hop_adjs, os.path.join(adjs_path, file_name))
        else:
            hop_adjs = torch.load(os.path.join(adjs_path, file_name))
        new_paths = list(hop_adjs.keys())
        print(f'after reduce redundancy, paths num: {len(paths)} => {len(new_paths)}')
        end = time.time()
        print(f'Time used for reduce redundancy: {end - start}')

    start = time.time()
    paths = list(hop_adjs.keys())
    if args.dataset not in ['AMiner']:
        feats = {tgt_type: feats_dict[tgt_type]}
        for path in paths:
            feats[path] = calc.matmul(hop_adjs[path], feats_dict[path[-1]])
        if args.sim_module:
            feature_sim = prune(get_feature_sim(feats_dict[tgt_type]), args.threshold1).to(device)
            jaccard_sim = get_jaccard_sim(dl).to(device)
            for path in paths:
                start_node, end_node = path[0], path[-1]
                if start_node != end_node:
                    src_idx = idx_shift_dict[start_node]
                    dst_idx = idx_shift_dict[end_node]
                    sim = prune(jaccard_sim[src_idx[0]: src_idx[1], dst_idx[0]: dst_idx[1]], args.threshold2)
                else:
                    sim = feature_sim
                if sim.nnz() > 0:
                    new_adj = calc.mul(hop_adjs[path], sim.fill_value(1.))
                    sim_feat = calc.matmul(new_adj, feats_dict[end_node])
                    feats[path] = feats[path] + args.alpha * sim_feat
    else:
        feats = {tgt_type: SparseTensor.eye(dl.nodes['count'][0])}
        for path in paths:
            feats[path] = hop_adjs[path]
        if args.sim_module:
            jaccard_sim = get_jaccard_sim(dl)
            for path in paths:
                start_node, end_node = path[0], path[-1]
                src_idx = idx_shift_dict[start_node]
                dst_idx = idx_shift_dict[end_node]
                sim = prune(jaccard_sim[src_idx[0]: src_idx[1], dst_idx[0]: dst_idx[1]], args.threshold2)
                if sim.nnz() > 0:
                    new_adj = calc.mul(hop_adjs[path], sim.fill_value(1.))
                    feats[path] = feats[path] + SparseTensor.from_dense(args.alpha * new_adj.to_dense())
    end = time.time()
    print(f'Time used for feature propagation: {end - start}')

    del hop_adjs
    torch.cuda.empty_cache()
    gc.collect()

    # =======
    # Train & eval loaders
    # =======
    train_loader = torch.utils.data.DataLoader(train_idx, batch_size=args.batch_size, shuffle=True, drop_last=False)

    eval_loader, full_loader = [], []
    eval_batch_size = 2 * args.batch_size

    for batch_idx in range((labeled_num_nodes - 1) // eval_batch_size + 1):
        batch_start = batch_idx * eval_batch_size
        batch_end = min(labeled_num_nodes, (batch_idx + 1) * eval_batch_size)
        batch = torch.LongTensor(labeled_idx[batch_start:batch_end])
        batch_feats = {k: x[batch] for k, x in feats.items()}
        eval_loader.append((batch, batch_feats))

    for batch_idx in range((len(extra_idx)-1) // eval_batch_size + 1):
        batch_start = batch_idx * eval_batch_size
        batch_end = min(len(extra_idx), (batch_idx+1) * eval_batch_size)
        batch = torch.LongTensor(extra_idx[batch_start:batch_end])
        batch_feats = {k: x[batch] for k, x in feats.items()}
        full_loader.append((batch, batch_feats))

    in_dims = {k: v.size(-1) for k, v in feats.items()}
    model = EMHGNN(args.dataset, in_dims, feats.keys(), args.embed_dim, args.hidden, num_classes, tgt_type,
                    args.input_drop, args.dropout, args.out_layers, args.residual, args.attention_module)
    model = model.to(device)
    labels_cuda = labels.to(device)
    if args.attention_module:
        attention_params = [model.length_attention, model.type_attention]
        other_params = list(set(model.parameters()) - set(attention_params))
        optimizer = torch.optim.Adam([
                {'params': other_params, 'lr': args.lr, 'weight_decay': args.wd},
                {'params': attention_params, 'lr': args.attn_lr, 'weight_decay': args.attn_wd},
            ])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.seed == args.seeds[0]:
        print(f'For tgt type {tgt_type}, feature keys (num={len(feats)}): {feats.keys()}')
        print('# Params:', get_n_params(model))

    if args.dataset == 'IMDB':
        loss_fcn = nn.BCEWithLogitsLoss()
    else:
        loss_fcn = nn.CrossEntropyLoss()
    if args.amp and not args.cpu:
        scalar = torch.cuda.amp.GradScaler()
    else:
        scalar = None
    early_stopping = EarlyStopping(patience=args.patience, save_path='output/checkpoint_{}.pt'.format(args.dataset))

    train_times = []
    for epoch in range(args.epoch):
        # training
        start = time.time()
        train(model, feats, labels_cuda, loss_fcn, optimizer, train_loader, scalar=scalar)
        end = time.time()
        train_times.append(end - start)

        # validation
        with torch.no_grad():
            model.eval()
            raw_pred = []
            for batch, batch_feats in eval_loader:
                batch_feats = {k: x.to(device) for k, x in batch_feats.items()}
                logits = model(batch_feats)
                raw_pred.append(logits.cpu())
            raw_pred = torch.cat(raw_pred, dim=0)
            val_loss = loss_fcn(raw_pred[train_val_point:val_test_point], labels[val_idx]).item()

            train_acc = evaluate(raw_pred[:train_val_point], labels[train_idx], args.dataset)
            val_acc = evaluate(raw_pred[train_val_point:val_test_point], labels[val_idx], args.dataset)
            test_acc = evaluate(raw_pred[val_test_point:labeled_num_nodes], labels[test_idx], args.dataset)
            if epoch % 10 == 0:
                print(f'epoch: {epoch}, val_loss: {val_loss:.4f}, '
                      f'train_acc: ({train_acc[0] * 100:.2f}, {train_acc[1] * 100:.2f}), '
                      f'val_acc: ({val_acc[0] * 100:.2f}, {val_acc[1] * 100:.2f}), '
                      f'test_acc: ({test_acc[0] * 100:.2f}, {test_acc[1] * 100:.2f})')

        # early stopping
        early_stopping(epoch, val_loss, val_acc, raw_pred, model, judge_loss=True, save_model=True)
        if early_stopping.early_stop:
            print('Early stopping!')
            break

    print(f'average train times: {sum(train_times) / len(train_times)}')
    times_list.append(sum(train_times) / len(train_times))
    all_pred = torch.empty((num_nodes, num_classes))
    all_pred[labeled_idx] = early_stopping.best_pred

    # weights
    # if args.attention_module:
    #     model.load_state_dict(torch.load(f'output/checkpoint_{args.dataset}.pt'))
    #     feats = {k: x.to(device) for k, x in feats.items()}
    #     weights = model.get_weights()
    #     weights_dict = {k: v for k, v in zip(feats.keys(), weights)}
    #     print(weights_dict)


    # if len(full_loader):
    #     model.load_state_dict(torch.load(f'output/checkpoint_{args.dataset}.pt'))
    #     with torch.no_grad():
    #         model.eval()
    #         raw_pred = []
    #         for batch, batch_feats in full_loader:
    #             batch_feats = {k: x.to(device) for k, x in batch_feats.items()}
    #             raw_pred.append(model(batch_feats).cpu())
    #         raw_pred = torch.cat(raw_pred, dim=0)
    #     all_pred[extra_idx] = raw_pred

    train_acc = evaluate(all_pred[train_idx], labels[train_idx], args.dataset)
    val_acc = evaluate(all_pred[val_idx], labels[val_idx], args.dataset)
    test_acc = evaluate(all_pred[test_idx], labels[test_idx], args.dataset)
    print(f'best_epoch: {early_stopping.best_epoch}, val_loss: {-early_stopping.best_score:.4f}', end=' ')
    print(f'train_acc: ({train_acc[0] * 100:.2f}, {train_acc[1] * 100:.2f}) '
          f'val_acc: ({val_acc[0] * 100:.2f}, {val_acc[1] * 100:.2f}) '
          f'test_acc: ({test_acc[0] * 100:.2f}, {test_acc[1] * 100:.2f})')
    print('=' * 110)
    return [test_acc[0] * 100, test_acc[1] * 100]


def parse_args(dataset):
    yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "args.yaml")
    parser = argparse.ArgumentParser(description='EMHGNN')
    # For environment costruction
    parser.add_argument('--seeds', nargs='+', type=int, default=[5, 6, 7, 8, 9])
    parser.add_argument('--dataset', type=str, default=dataset, choices=['DBLP', 'ACM', 'IMDB', 'AMiner'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cpu', type=bool, default=False)
    parser.add_argument('--root', type=str, default='./data/')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--embed-dim', type=int, default=512)
    parser.add_argument('--num-hops', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--threshold1', type=float, default=0.5)
    parser.add_argument('--threshold2', type=float, default=0.5)
    parser.add_argument('--ratio', type=str, default='0.6,0.9')
    parser.add_argument('--ACM-keep-F', type=bool, default=False)
    parser.add_argument('--test', action='store_true', default=False)

    # For network structure
    parser.add_argument('--out-layers', type=int, default=3)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--input-drop', type=float, default=0.5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--residual', action='store_true', default=False)

    # for training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--attn-lr', type=float, default=0.002)
    parser.add_argument('--attn-wd', type=float, default=0)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=10000)
    parser.add_argument('--amp', action='store_true', default=True)
    # for ablation study
    parser.add_argument('--attention-module', type=bool, default=True)
    parser.add_argument('--redundancy-module', type=bool, default=True)
    parser.add_argument('--sim-module', type=bool, default=True)
    with open(yaml_path) as args_file:
        try:
            parser.set_defaults(**dict(YAML().load(args_file)[dataset].items()))
        except KeyError:
            raise AssertionError("KeyError: there's no {} in yamls".format(dataset), "red")
    return parser.parse_args()


if __name__ == '__main__':
    dataset = sys.argv[2]
    args = parse_args(dataset)
    print(args)

    results = []
    logging.basicConfig(filename=f'log/{args.dataset}.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    for seed in args.seeds:
        args.seed = seed
        print('Restart with seed =', seed)
        result = main(args)
        results.append(result)

    print('results', results)
    aver = list(map(list, zip(*results)))
    print(f'macro_aver: {np.mean(aver[0]):.2f}', f'macro_std: {np.std(aver[0]):.2f}')
    print(f'micro_aver: {np.mean(aver[1]):.2f}', f'micro_std: {np.std(aver[1]):.2f}')
    if args.test:
        logging.info(f'macro_aver: {np.mean(aver[0]):.2f}, macro_std: {np.std(aver[0]):.2f}')
        logging.info(f'micro_aver: {np.mean(aver[1]):.2f}, micro_std: {np.std(aver[1]):.2f}')
    print(f'all average train times: {sum(times_list) / len(times_list)}')
