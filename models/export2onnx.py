"""Exports a YOLOv5 *.pt model to ONNX and TorchScript formats

Usage:
    $ export PYTHONPATH="$PWD" && python models/export.py --weights ./weights/yolov5s.pt --img 640 --batch 1
"""
import sys
sys.path.append('./')  # to run '$ python *.py' files in subdirectories
from  utils.activations import Hardswish, SiLU
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
import models
import torch.nn as nn
import torch
import argparse

import time
from PIL import Image
import numpy as np

torch.set_printoptions(precision=5)



def pad_image2(im, max_size=640):
    try:
        width, height = im.size
        max_length = max(width, height)
        new_im = Image.new("RGB", (max_size, max_size))
        if width > height:
            new_width = max_size
            new_height = int(new_width*height/width)
            top_x = 0
            top_y = (new_width - new_height)//2
            im_resized = im.resize(
                (new_width, new_height), resample=Image.BILINEAR)
            new_im.paste(im_resized, (top_x, top_y))
        else:
            new_height = max_size
            new_width = int(new_height*width/height)
            im_resized = im.resize(
                (new_width, new_height), resample=Image.BILINEAR)
            top_x = (new_height - new_width)//2
            top_y = 0
            new_im.paste(im_resized, (top_x, top_y))
        # else:
        #     new_im = im
        return new_im
    except Exception as e:
        print(e.message)
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov5s.pt',
                        help='weights path')  # from yolov5/models/
    parser.add_argument('--img-size', nargs='+', type=int,
                        default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--output_name', type=str, default='')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    set_logging()
    t = time.time()

    # Load PyTorch model
    model = attempt_load(
        opt.weights, map_location=torch.device('cpu'))  # load FP32 model
    labels = model.names

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    # verify img_size are gs-multiples
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]

    # Input
    # image size(1,3,320,192) iDetection
    img = torch.randn(opt.batch_size, 3, *opt.img_size)
    # image = np.asarray(pad_image2(Image.open('test2.jpg')),
    #                    dtype=np.float32)/255.0
    # img = torch.from_numpy(np.expand_dims(image.transpose(2, 0, 1), 0))

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        # elif isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)
    model.model[-1].export = True
    model.eval()
    print(model.model[-1].export)
    y = model(img)  # dry run

    # print(model.model[-1].anchors)
    # print(model.model[-1].anchor_grid)
    # print(model.model[-1].stride)
    # print(f'batchsize: {len(y)}')
    # pred = y[0]
    # print(y.shape)
    # print(y[0, 0, :10])
    # print(y[0, 0, :10])
    # print(y[0, -1, :10])
    # for i, o in enumerate(y):
    #     print(i, o.shape)
    # print(y[0][0, 310:315, :5])

    # det = non_max_suppression(pred, 0.25, 0.45)
    # result = det[0].numpy()
    # for x, y, w, h, conf, cls in result:
    #     print(x, y, w, h, conf, cls)

    # ONNX export
    try:
        import onnx
        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        if opt.output_name:
            onnx_model_name = opt.output_name
        else:
            onnx_model_name = opt.weights.replace('.pt', '.onnx')  # filename
        torch.onnx.export(model, img, onnx_model_name, verbose=False,
                          opset_version=12, input_names=['images'],
                          output_names=['p3', 'p4', 'p5'])

        # Checks
        onnx_model = onnx.load(onnx_model_name)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        print('ONNX export success, saved as %s' % onnx_model_name)
    except Exception as e:
        print('ONNX export failure: %s' % e)

    # Finish
    print('\nExport complete (%.2fs).' % (time.time() - t))
