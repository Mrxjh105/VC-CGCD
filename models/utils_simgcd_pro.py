from copy import deepcopy
import numpy as np
from sklearn.cluster import KMeans
import torch
from tqdm import tqdm
import faiss

'''
get_kmeans_centroid_for_new_head(): use model_pre to obtain new class head init

compute_prior_old_new_ratio(): use model_cur to predict old and new ratio

both use online session training data of current stage
'''


def get_kmeans_centroid_for_new_head(model, online_session_train_loader, args, device):
    """
    Extract features from the current online session data, run faiss-kmeans

    Returns
    -------
    new_head : torch.Tensor               # (args.num_novel_class_per_session, feat_dim)
    results  : dict                       # {'im2cluster':[...], 'centroids':[...], 'density':[...]}
    """

    model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # 1. Collect features from the backbone
    # ------------------------------------------------------------------
    all_feats = []
    args.logger.info('Perform KMeans for new classification head initialization!')
    args.logger.info('Collating features...')
    with torch.no_grad():
        for images, *_ in tqdm(online_session_train_loader, desc='Feature extraction'):
            images = images.cuda(device, non_blocking=True)
            feats = model[0](images)          # backbone
            feats = torch.nn.functional.normalize(feats, dim=-1)
            all_feats.append(feats.cpu().numpy())

    feats_np = np.concatenate(all_feats, axis=0).astype('float32')   # (N, D)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 2. Run faiss-kmeans
    # ------------------------------------------------------------------
    # We cluster into (labelled + current novel) clusters
    n_clusters = args.num_labeled_classes + args.num_cur_novel_classes

    d = feats_np.shape[1]
    k = int(n_clusters)

    # faiss gpu clustering setup
    clus = faiss.Clustering(d, k)
    clus.verbose = False
    clus.niter = 20
    clus.nredo = 5
    clus.seed = 0
    clus.max_points_per_centroid = 1000
    clus.min_points_per_centroid = 10

    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = device.index if torch.cuda.is_available() else 0
    index = faiss.GpuIndexFlatL2(res, d, cfg)

    clus.train(feats_np, index)

    # cluster assignments
    D, I = index.search(feats_np, 1)          # (N, 1)
    im2cluster = [int(n[0]) for n in I]

    # centroids
    centroids_np = faiss.vector_to_array(clus.centroids).reshape(k, d)

    # compute per-cluster densities
    Dcluster = [[] for _ in range(k)]
    for im, i in enumerate(im2cluster):
        Dcluster[i].append(D[im][0])

    density = np.zeros(k)
    for i, dist_list in enumerate(Dcluster):
        if len(dist_list) > 1:
            d_avg = (np.asarray(dist_list, dtype=np.float32)**0.5).mean() / np.log(len(dist_list) + 10)
            density[i] = d_avg
    dmax = density.max()
    for i, dist_list in enumerate(Dcluster):
        if len(dist_list) <= 1:
            density[i] = dmax

    # clamp extreme values for stability
    low, high = np.percentile(density, 10), np.percentile(density, 90)
    density = density.clip(low, high)
    density = args.temperature * density / density.mean()   # scale mean to temperature

    # convert to torch cuda tensors
    centroids = torch.from_numpy(centroids_np).cuda(device)
    centroids = torch.nn.functional.normalize(centroids, dim=1)      # (k, D)
    im2cluster = torch.LongTensor(im2cluster).cuda(device)
    density = torch.Tensor(density).cuda(device)

    # package the same results dict as run_kmeans
    results = {
        'im2cluster': [im2cluster],
        'centroids':  [centroids],
        'density':    [density]
    }
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 3. Recover new_head
    # ------------------------------------------------------------------
    with torch.no_grad():
        _, logits = model[1](centroids)           # (k, num_seen_classes)
        max_logits, _ = torch.max(logits, dim=-1) # (k,)
        _, proto_idx = torch.topk(max_logits,
                                  k=args.num_novel_class_per_session,
                                  largest=False)  # pick the smallest
        new_head = centroids[proto_idx]           # (num_novel, D)
    # ------------------------------------------------------------------

    return new_head, results


def compute_prior_old_new_ratio(model, online_session_train_loader, args, device):
    model.to(device)
    model.eval()

    all_preds_list = [] = []
    args.logger.info('Using model_cur (initialized new heads) to predicting labels...')
    # First extract all features
    with torch.no_grad():
        for batch_idx, (images, label, _, _) in enumerate(tqdm(online_session_train_loader)):
            images = images.cuda(non_blocking=True)
            _, logits = model(images)
            all_preds_list.append(logits.argmax(1))

    all_preds = torch.cat(all_preds_list, dim=0)   # NOTE!!!
    args.logger.info('Computing prior old and new ratio...')
    pred_prior_old_ratio = len(all_preds[all_preds<args.num_seen_classes]) / len(all_preds)
    pred_prior_new_ratio = len(all_preds[all_preds>=args.num_seen_classes]) / len(all_preds)
    args.logger.info(f'Pred prior old ratio: {pred_prior_old_ratio:.4f} | Pred prior new ratio: {pred_prior_new_ratio:.4f}')


    pred_prior_ratio_dict = {
        'pred_prior_old_ratio': pred_prior_old_ratio,
        'pred_prior_new_ratio': pred_prior_new_ratio,
    }

    return pred_prior_ratio_dict
