import argparse
# import json
import numpy as np
import absl.logging as log
log.set_verbosity(log.INFO)
from torch.utils.data import DataLoader
from models import *
from utils.datasets import *
from utils.utils import *
from py.indicators import *


def test(
    cfg,
    data,
    weights=None,
    batch_size=16,
    imgsz=416,
    conf_thres=0.001,
    iou_thres=0.6,
    save_json=False,
    single_cls=False,
    augment=False,
    model=None,
    dataloader=None,
    multi_label=True
):
    """TODO

    Args:
        TODO

    Raises:
        TODO

    Returns:
        TODO
    """
    if model is None:
        is_training = False
        device = torch_utils.select_device(opt.device, batch_size=batch_size)
        verbose = opt.task == 'test'
        for f in glob.glob('test_batch*.jpg'):
            os.remove(f)
        model = Darknet(cfg, imgsz)
        attempt_download(weights)
        if weights.endswith('.pt'):  
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  
            load_darknet_weights(model, weights)
        # Fuse
        model.fuse()
        model.to(device)
        if device.type != 'cpu' and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:  # called by train.py
        is_training = True
        device = next(model.parameters()).device  
        verbose = False
    # Configure run
    data = parse_data_cfg(data)
    nc = 1 if single_cls else int(data['classes'])  
    path = data['valid']  
    names = load_classes(data['names'])  
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    iouv = iouv[0].view(1)  # comment for mAP@0.5:0.95
    iouv = torch.Tensor([0.35]).to(device) # adaption
    niou = iouv.numel()
    # Dataloader
    if dataloader is None:
        dataset = ImagesAndLabelsLoader(path, imgsz, batch_size, single_cls=opt.single_cls, pad=0.5)
        batch_size = min(batch_size, len(dataset))
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
            pin_memory=True,
            collate_fn=dataset.collate_fn
        )
    seen = 0
    model.eval()
    _ = model(torch.zeros((1, 3, imgsz, imgsz), device=device)) if device.type != 'cpu' else None  # run once
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%10s' * 9) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1', 'derror', 'dacc', 'dstd')
    p, r, f1, mp, mr, map, mf1, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    d_error, d_acc = [], [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    # depth_stats = [[],[],[],[],[]] # adaption, for saving depth error stats with 5 different ranges(in merters):[0,10]/[10,30]/[30/50]/[50,80]/[80,150].
    de_acc = [[],[],[],[],[]]
    for batch_i, (imgs, targets, paths, shapes, roi_info) in enumerate(tqdm(dataloader, desc=s)): # adaption, original:"for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):"
        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = imgs.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)
        # Disable gradients
        with torch.no_grad():
            # Run model
            t = torch_utils.time_synchronized()
            inf_out, train_out, roidepth_out = model(imgs, roi=roi_info, augment=augment)  # inference and training outputs. adaption, original:"inf_out, train_out, roidepth_out = model(imgs, augment=augment)"
            t0 += torch_utils.time_synchronized() - t
            # Compute loss
            if is_training:  # if model has loss hyperparameters
                loss += compute_loss(train_out, roidepth_out, targets, model)[1][:3]  # GIoU, obj, cls
            # Run NMS
            t = torch_utils.time_synchronized()
            output = non_max_suppression(
                inf_out, conf_thres=conf_thres, iou_thres=iou_thres, multi_label=multi_label
            )
            t1 += torch_utils.time_synchronized() - t
        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1
            if pred is None:
                if nl:
                    stats.append(
                        (torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls)
                    )
                continue
            # Append to text file
            # with open('test.txt', 'a') as file:
            #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))
            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(Path(paths[si]).stem.split('_')[-1])
                box = pred[:, :4].clone()  # xyxy
                scale_coords(imgs[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                box = xyxy_to_xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append(
                        {
                            'image_id': image_id,
                            'category_id': coco91class[int(p[5])],
                            'bbox': [round(x, 3) for x in b],
                            'score': round(p[4], 5)
                        }
                    )
            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]
                # target boxes
                tbox = xywh_to_xyxy(labels[:, 1:5]) * whwh
                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # target indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # prediction indices
                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices
                        # Append detections
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break
            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
            # Depth stats for eval, adaption.
            tdepth = labels[:,5]
            pdepth, tdepth = np.squeeze(
                roidepth_out.cpu().numpy()
            )*Y_RANGE, tdepth.cpu().numpy()*Y_RANGE
            # de = abs(tdepth-pdepth)
            # if 1<tdepth<=10: # 0~10m
            #     depth_stats[0].append(de)
            #     de_acc[0].append(1-de/tdepth)
            # elif 10<tdepth<=30: # 10~30m
            #     depth_stats[1].append(de)
            #     de_acc[1].append(1-de/tdepth)
            # elif 30<tdepth<=50: # 30~50m
            #     depth_stats[2].append(de)
            #     de_acc[2].append(1-de/tdepth)
            # elif 50<tdepth<=80: # 50~80m
            #     depth_stats[3].append(de)
            #     de_acc[3].append(1-de/tdepth)
            # else:# 80~150m
            #     depth_stats[4].append(de)
            #     de_acc[4].append(1-de/tdepth)

            collect_depth(d_error, d_acc, tdepth, pdepth)
        # Plot images
        if batch_i < 1:
            f = 'test_batch%g_gt.jpg' % batch_i  # filename
            plot_images(imgs, targets, paths=paths, names=names, fname=f)  # ground truth
            f = 'test_batch%g_pred.jpg' % batch_i
            plot_images(
                imgs, output_to_target(output, width, height), paths=paths, names=names, fname=f
            )  # predictions
    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        if niou > 1:
            p, r, ap, f1 = p[:, 0], r[:, 0], ap.mean(1), ap[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Eval depth, adaption 
    # de_mean = [np.mean(ds) for ds in depth_stats] # get mean error in different range
    # de_acc = [np.mean(da) for da in de_acc] # get accuracy error in different range
    # de_std = [np.std(me) for me in de_mean] # get standard deviation of depth error in different range
    # len_stats = [len(depth_stats[0]), len(depth_stats[1]), len(depth_stats[2]), len(depth_stats[3]), len(depth_stats[4])]
    # total_mean = sum([x*y if y>0 else 0 for x,y in zip(de_mean, len_stats)])/sum(len_stats)
    # total_acc = sum([x*y if y>0 else 0 for x,y in zip(de_acc, len_stats)])/sum(len_stats)
    # total_std = sum([x*y if y>0 else 0 for x,y in zip(de_std, len_stats)])/sum(len_stats)
    # depth_stats = [str(len_stats)+'\n', str(de_mean)+'\n', 
    #                 str(de_acc)+'\n', str(de_std)+'\n']
    # with open('depth-test-stats.txt', 'w', encoding='utf-8') as f:
    #     f.writelines(depth_stats)

    cal_depth_indicators(d_error, d_acc)
    # Print results
    pf = '%20s' + '%10.3g' * 6  
    log.info(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))
    # Print results per class
    if verbose and nc > 1 and len(stats):
        pfc = '%20s' + '%10.3g' * 6  # print format
        for i, c in enumerate(ap_class):
            log.info(pfc % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))
    # Print speeds
    if verbose or save_json:
        t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
        log.info('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # # Save JSON
    # if save_json and map and len(jdict):
    #     log.info('\nCOCO mAP with pycocotools...')
    #     imgIds = [int(Path(x).stem.split('_')[-1]) for x in dataloader.dataset.img_files]
    #     with open('results.json', 'w') as file:
    #         json.dump(jdict, file)

    #     try:
    #         from pycocotools.coco import COCO
    #         from pycocotools.cocoeval import COCOeval

    #         # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    #         cocoGt = COCO(glob.glob('../coco/annotations/instances_val*.json')[0])  # initialize COCO ground truth api
    #         cocoDt = cocoGt.loadRes('results.json')  # initialize COCO pred api

    #         cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    #         cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
    #         cocoEval.evaluate()
    #         cocoEval.accumulate()
    #         cocoEval.summarize()
    #         # mf1, map = cocoEval.stats[:2]  # update to pycocotools results (mAP@0.5:0.95, mAP@0.5)
    #     except:
    #         log.info('WARNING: pycocotools must be installed with numpy==1.17 to run correctly. '
    #               'See https://github.com/cocodataset/cocoapi/issues/356')

    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map, mf1, *(loss.cpu() / len(dataloader)).tolist()), maps

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--cfg', type=str, default='cfg/roidepth_0_0_2.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/roidepth-kitti.data', help='*.data path')
    parser.add_argument('--weights', type=str, default='weights/last.pt', help='weights path')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=128, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.35, help='IOU threshold for NMS')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='test', help="'test', 'study', 'benchmark'")
    parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.save_json = opt.save_json or any([x in opt.data for x in ['coco.data', 'coco2014.data', 'coco2017.data']])
    opt.cfg = check_file(opt.cfg)  
    opt.data = check_file(opt.data)  
    log.info(opt)
    Y_RANGE=100
    # task = 'test', 'study', 'benchmark'
    if opt.task == 'test':
        test(
            opt.cfg,
            opt.data,
            opt.weights,
            opt.batch_size,
            opt.img_size,
            opt.conf_thres,
            opt.iou_thres,
            opt.save_json,
            opt.single_cls,
            opt.augment
        )
    elif opt.task == 'benchmark':  # mAPs at 256-640 at conf 0.5 and 0.7
        y = []
        for i in list(range(256, 640, 128)):
            for j in [0.6, 0.7]:  # iou-thres
                t = time.time()
                r = test(
                    opt.cfg, opt.data, 
                    opt.weights, opt.batch_size, 
                    i, opt.conf_thres, 
                    j, opt.save_json
                )[0]
                y.append(r + (time.time() - t,))
        np.savetxt('benchmark.txt', y, fmt='%10.4g') 
