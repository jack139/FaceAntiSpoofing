import os
from random import randint
import torch
import torchvision
from trainer.base import BaseTrainer
from utils.meters import AvgMeter
from utils.eval import add_visualization_to_tensorboard, predict, calc_accuracy
from tqdm import tqdm

PREDICT_THRESHOLD = 0.45

class FASTrainer(BaseTrainer):
    def __init__(self, cfg, network, optimizer, criterion, lr_scheduler, device, trainloader, valloader, writer):
        super(FASTrainer, self).__init__(cfg, network, optimizer, criterion, lr_scheduler, device, trainloader, valloader, writer)

        self.network = self.network.to(device)

        self.train_loss_metric = AvgMeter(writer=writer, name='Loss/train', num_iter_per_epoch=len(self.trainloader), per_iter_vis=True)
        self.train_acc_metric = AvgMeter(writer=writer, name='Accuracy/train', num_iter_per_epoch=len(self.trainloader), per_iter_vis=True)

        self.val_loss_metric = AvgMeter(writer=writer, name='Loss/val', num_iter_per_epoch=len(self.valloader))
        self.val_acc_metric = AvgMeter(writer=writer, name='Accuracy/val', num_iter_per_epoch=len(self.valloader))

        self.best_val_acc = 0.0

    def load_model(self, ckpt_name):
        saved_name = os.path.join(self.cfg['output_dir'], ckpt_name)
        state = torch.load(saved_name, map_location=self.device)

        self.optimizer.load_state_dict(state['optimizer'])
        self.network.load_state_dict(state['state_dict'])


    def save_model(self, epoch, epoch_acc):
        if not os.path.exists(self.cfg['output_dir']):
            os.makedirs(self.cfg['output_dir'])

        saved_name = os.path.join(self.cfg['output_dir'], 
            '{}_{}_e{}_acc_{:.4f}.pth'.format(self.cfg['model']['base'], self.cfg['dataset']['name'], epoch, epoch_acc))

        state = {
            'epoch': epoch,
            'state_dict': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        
        torch.save(state, saved_name)


    def train_one_epoch(self, epoch):
        self.network.train()
        self.train_loss_metric.reset(epoch)
        self.train_acc_metric.reset(epoch)

        print('\nEpoch: {}, lr: {}'.format(epoch+1, self.lr_scheduler.get_last_lr()))
        for i, (img, depth_map, rppg, label) in tqdm(enumerate(self.trainloader), total=len(self.trainloader)):
            img, depth_map, rppg, label = img.to(self.device), depth_map.to(self.device), \
                rppg.type(torch.FloatTensor).to(self.device), label.to(self.device)
            rppg_depth, net_depth_map, _, _, _, _, _ = self.network(img, rppg)
            mix_depth = (net_depth_map + rppg_depth) / 2 # 算术平均
            self.optimizer.zero_grad()
            #loss = self.criterion(net_depth_map, rppg_depth, depth_map)
            loss = self.criterion(mix_depth, depth_map)
            loss.backward()
            self.optimizer.step()

            preds, _ = predict(mix_depth, threshold=PREDICT_THRESHOLD)
            targets, _ = predict(depth_map, threshold=PREDICT_THRESHOLD)

            accuracy = calc_accuracy(preds, targets)

            # Update metrics
            self.train_loss_metric.update(loss.item())
            self.train_acc_metric.update(accuracy)

        print('iter: {}, loss: {:.6f}, acc: {:.4f}'.format(epoch * len(self.trainloader) \
            + i, self.train_loss_metric.avg, self.train_acc_metric.avg))


    def train(self):

        for epoch in range(self.cfg['train']['num_epochs']):
            self.train_one_epoch(epoch)
            epoch_acc = self.validate(epoch)
            if epoch_acc >= self.best_val_acc:
                self.best_val_acc = epoch_acc
                self.save_model(epoch, epoch_acc)
            print('val_acc: {:.4f}, best_val_acc: {:.4f}'.format(epoch_acc, self.best_val_acc))
            self.lr_scheduler.step()

    def validate(self, epoch):
        import numpy as np
        from sklearn.metrics import f1_score, accuracy_score

        self.network.eval()
        self.val_loss_metric.reset(epoch)
        self.val_acc_metric.reset(epoch)

        Xscore = []
        Ylabel = []

        seed = randint(0, len(self.valloader)-1)
        with torch.no_grad():
            for i, (img, depth_map, rppg, label) in tqdm(enumerate(self.valloader), total=len(self.valloader)):
                img, depth_map, rppg, label = img.to(self.device), depth_map.to(self.device), \
                    rppg.type(torch.FloatTensor).to(self.device), label.to(self.device)
                rppg_depth, net_depth_map, _, _, _, _, _ = self.network(img, rppg)
                mix_depth = (net_depth_map + rppg_depth) / 2 # 算术平均
                #loss = self.criterion(net_depth_map, rppg_depth, depth_map)
                loss = self.criterion(mix_depth, depth_map)
                
                preds, score = predict(mix_depth, threshold=PREDICT_THRESHOLD)
                targets, _ = predict(depth_map, threshold=0.011)
                #print(preds, targets, score)
                accuracy = calc_accuracy(preds, targets)

                # Update metrics
                self.val_loss_metric.update(loss.item())
                self.val_acc_metric.update(accuracy)

                if i == seed:
                    add_visualization_to_tensorboard(self.cfg, epoch, img, preds, targets, score, self.writer)

                Xscore.extend(score.tolist())
                Ylabel.extend(targets.tolist())

            Xscore = np.array(Xscore)
            Ylabel = np.array(Ylabel)

            thresholds = np.arange(0.1, 1.0, 0.01)

            f1_scores = [f1_score(Ylabel, Xscore >= t) for t in thresholds]
            acc_scores = [accuracy_score(Ylabel, Xscore >= t) for t in thresholds]

            opt_idx = np.argmax(f1_scores)
            # Threshold at maximal F1 score
            opt_tau = thresholds[opt_idx]
            # Accuracy at maximal F1 score
            opt_f1 = f1_score(Ylabel, Xscore >= opt_tau)
            opt_acc = accuracy_score(Ylabel, Xscore >= opt_tau)

            #print("thresholds: ", thresholds)
            #print("f1_scores: ", f1_scores)
            #print("acc_scores: ", acc_scores)
            print("opt_threshold: %.4f"%opt_tau)
            print("f1: %.4f\tacc: %.4f"%(opt_f1, opt_acc))

            #return self.val_acc_metric.avg
            return opt_acc
