import train
import train_dual


class Yolov9:

    def __init__(self):
        pass

    def train(self, epochs=1, data=None, device=0, mode='dual'):

        assert data is not None
        assert mode in ['dual', 'gelan']

        if mode in ['dual']:
            opt = train_dual.parse_opt()

            opt.epochs = epochs
            opt.data = data

            opt.workers = 8
            opt.device = device
            opt.batch = 8
            opt.img = 320
            opt.cfg = 'models/detect/yolov9-c.yaml'
            opt.weights = ''
            opt.name = 'yolov9-c'
            opt.hyp = 'hyp.scratch-high.yaml'
            opt.min_items = 0
            opt.close_mosaic = 15

            train_dual.main(opt)

        elif mode in ['gelan']:

            opt = train.parse_opt()

            opt.epochs = epochs
            opt.data = data

            opt.workers = 8
            opt.device = device
            opt.batch = 32
            opt.img = 640
            opt.cfg = 'models/detect/gelan-c.yaml'
            opt.weights = ''
            opt.name = 'gelan-c'
            opt.hyp = 'hyp.scratch-high.yaml'
            opt.min_items = 0
            opt.close_mosaic = 15

            train.main(opt)

    def eval(self, data=None):
        pass

    def predict(self, image=None):
        pass
