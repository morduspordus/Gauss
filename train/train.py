# Modified from segmentation-models-pytorch V 0.0.3

import sys
import torch
from tqdm import tqdm as tqdm
from segmentation_models_pytorch.utils.meter import AverageValueMeter
import torch.nn as nn


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
      m.eval()


class Epoch:

    def __init__(self, model, evaluator, losses, stage_name,  device='cpu', verbose=True):
        self.model = model
        self.losses = losses
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self._to_device()
        self.evaluator = evaluator

    def _to_device(self):
        for loss in self.losses:
            loss.to(self.device)
        self.model.to(self.device)

    @classmethod
    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def on_epoch_end(self):
        pass



    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        losses_meters = {loss.__name__: AverageValueMeter() for loss in self.losses}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for sample in iterator:
                x = sample['image']
                y = sample['label']

                x, y = x.to(self.device), y.to(self.device)

                loss, losses, y_pred = self.batch_update(x, y, sample)

                # update losses logs

                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {'total_loss': loss_meter.mean}
                logs.update(loss_logs)

                for k, v in losses.items():
                    if type(v) == torch.Tensor:
                        v = v.cpu().detach().numpy()
                    losses_meters[k].add(v)

                losses_logs = {k: v.mean for k, v in losses_meters.items()}
                logs.update(losses_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        logs['metrics'] = self.evaluator.compute_all_metrics()

        self.on_epoch_end()

        return logs


class TrainEpoch(Epoch):

    def __init__(self, model, evaluator, losses, optimizer, scheduler, args):
        super().__init__(
            model=model,
            evaluator=evaluator,
            losses=losses,
            stage_name='train',
            device=args['device'],
            verbose=args['verbose']
        )
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args

    def on_epoch_start(self):
        self.model.train()
        self.model.apply(set_bn_eval)
        self.evaluator.reset()

    def on_epoch_end(self):
        self.scheduler.step()

    def batch_update(self, x, y, sample):

        self.optimizer.zero_grad()
        prediction = self.model.forward(x)

        losses = {loss.__name__: loss(prediction, y, sample) for loss in self.losses}

        loss = 0.0

        for (k, v) in losses.items():
            loss = loss + v


        loss.backward()

        self.optimizer.step()

        self.evaluator.add_batch(y, prediction)

        return loss, losses, prediction


class ValidEpoch(Epoch):

    def __init__(self, model, evaluator, losses, device='cpu', verbose=True, imageSaver=None):
        super().__init__(
            model=model,
            evaluator=evaluator,
            losses=losses,
            stage_name='valid',
            device=device,
            verbose=verbose)

        self.imageSaver = imageSaver


    def on_epoch_start(self):
       self.model.eval()
       self.evaluator.reset()

    def batch_update(self, x, y, sample):
        with torch.no_grad():
            prediction = self.model.forward(x)

            losses = {loss.__name__: loss(prediction, y, sample) for loss in self.losses}

            loss = 0.0
            for (k, v) in losses.items():
                loss = loss + v

            self.evaluator.add_batch(y, prediction)

            if self.imageSaver is not None:
                self.imageSaver.save(prediction, y, x, sample['name'], sample['orig_size'])

        return loss, losses, prediction


