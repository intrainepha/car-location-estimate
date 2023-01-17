import argparse
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import absl.logging as log
log.set_verbosity(log.INFO)


def detect(save_img=False):
    """TODO

    Args:
        TODO

    Raises:
        TODO

    Returns:
        TODO
    """
    imgsz = (320, 192) if ONNX_EXPORT else opt.img_size  
    out, source, weights, half, view_img, save_txt = \
        opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') \
        or source.startswith('http') or source.endswith('.txt')
    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  
    os.makedirs(out)  
    # Initialize model
    model = Darknet(opt.cfg, imgsz)
    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)
    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])
        modelc.to(device).eval()
    model.to(device).eval()
    if ONNX_EXPORT:
        model.fuse()
        img = torch.zeros((1, 3) + imgsz)  # (1, 3, 320, 192)
        f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')  # *.onnx filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=11,
                          input_names=['images'], output_names=['classes', 'boxes'])
        # Validate exported model
        import onnx
        model = onnx.load(f)  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        log.info(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        return
    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()
    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = StreamsLoader(source, img_size=imgsz)
    else:
        save_img = True
        dataset = ImagesLoader(source, img_size=imgsz)
    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    badCnt = [0,0,0,0,0,0,0,0,0,0]
    for path, img, im0s, vid_cap in dataset:
        # if img.shape[1]/img.shape[2] <= 0.5:
        #     continue
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Get roi-info, adaption.
        label_path = path.replace('images', 'labels').replace('.jpg', '.txt')
        with open(label_path, 'r', encoding='utf-8') as lf:
            lines = lf.readlines()
            tdepth = [float(l.split(' ')[5]) for l in lines]
            roi_info = [[float(info) for info in l.split(' ')[7:11]] for l in lines]
        # Inference
        t1 = torch_utils.time_synchronized()
        pred, _, pdepth = model(img, roi=roi_info, augment=opt.augment)
        t2 = torch_utils.time_synchronized()
        # to float
        if half:
            pred = pred.float()
        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres,
            multi_label=False,
            classes=opt.classes,
            agnostic=opt.agnostic_nms
        )
        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        # Process detections
        for i, det in enumerate(pred):  # detections for image i
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s
            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy_to_xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                    if save_img or view_img:  # Add bbox to image
                        # label = '%s %.2f %.1f %.1f' % (names[int(cls)], conf, tdepth[i]*Y_RANGE, pdepth[i][0]*Y_RANGE)
                        label = '%.1f %.1f' % (tdepth[i]*Y_RANGE, pdepth[i][0]*Y_RANGE) # adaption
                        img_show, xyxy_show = resieze_img_and_box(im0, xyxy, minPix=100) # adaption
                        plot_one_box(xyxy_show, img_show, label=label, color=colors[int(cls)])
            # Print time (inference + NMS)
            log.info('%sDone. (%.3fs)' % (s, t2 - t1))
            # Stream results
            if view_img:
                cv2.imshow(p, img_show)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration
            # Save results (image with detections)
            if save_img:
                # if abs(tdepth[i]*Y_RANGE - pdepth[i][0]*Y_RANGE) < 5: continue # adaption
                if abs(tdepth[i] - pdepth[i][0])/tdepth[i] < 0.1: continue # adaption
                if 0<=tdepth[i]*Y_RANGE<10: badCnt[0] = badCnt[0]+1 
                if 10<=tdepth[i]*Y_RANGE<20: badCnt[1] = badCnt[1]+1 
                if 20<=tdepth[i]*Y_RANGE<30: badCnt[2] = badCnt[2]+1 
                if 30<=tdepth[i]*Y_RANGE<40: badCnt[3] = badCnt[3]+1 
                if 40<=tdepth[i]*Y_RANGE<50: badCnt[4] = badCnt[4]+1 
                if 50<=tdepth[i]*Y_RANGE<60: badCnt[5] = badCnt[5]+1 
                if 60<=tdepth[i]*Y_RANGE<70: badCnt[6] = badCnt[6]+1 
                if 70<=tdepth[i]*Y_RANGE<80: badCnt[7] = badCnt[7]+1 
                if 80<=tdepth[i]*Y_RANGE<90: badCnt[8] = badCnt[8]+1 
                if 90<=tdepth[i]*Y_RANGE<100: badCnt[9] = badCnt[9]+1 
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, img_show)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)
    if save_txt or save_img:
        log.info('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)
    log.info('Done. (%.3fs)' % (time.time() - t0))
    log.info(badCnt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/roidepth_0_0_2.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/cls5.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/last.pt', help='weights path')
    parser.add_argument('--source', type=str, default='/home/huyu/dataset/fv1xm/roi-boyue-test-tmp/car/images', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='boyue-output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=128, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.35, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', default=False, help='augmented inference')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.names = check_file(opt.names)  # check file
    log.info(opt)
    Y_RANGE = 100
    with torch.no_grad():
        detect()

