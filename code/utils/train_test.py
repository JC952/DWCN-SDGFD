from sklearn.metrics import accuracy_score, f1_score
import logging
import os
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
import models




class train_test(object):
    def __init__(self, args):
        self.args = args
    def setup(self,n_class):
        """
        Initialize the datasets, model, loss and optimizer
        """
        args = self.args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name=args.model_name
        self.model =getattr(models,self.model_name)(in_channel=1,num_classes=n_class,lr=args.lr).to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        return

    def train(self, op_num, source_train_loader, source_test_loader, target_test_loader, s_domain):
        best_acc = 0.0
        time_list = []
        if self.model_name in ['DWCN']:
            for epoch in range(self.args.epoch):
                epoch_start = time.time()
                # ======== Training phase ========
                loss_log = self.model(source_train_loader)
                elapsed_time = time.time() - epoch_start
                time_list.append(elapsed_time)
                logging.info('Num-{}, Epoch: {} Loss_c: {:.4f}, Loss_1: {:.4f}, Loss_2: {:.4f}, Time {:.4f} sec'.format(
                        op_num, epoch, loss_log["loss_c"], loss_log["loss_1"], loss_log["loss_2"],
                        time.time() - epoch_start))
                # ======== Source-domain evaluation ========
                self.model.eval()
                total_correct_s, total_samples_s = 0, 0
                with torch.no_grad():
                    for batch_idx, (inputs, labels, domain) in enumerate(source_test_loader):
                        labels = labels.to(self.device)
                        inputs = inputs.to(self.device)
                        logits = self.model.model_inference(inputs)
                        batch_size = inputs.size(0)
                        predict = logits.argmax(dim=1)
                        total_correct_s += torch.eq(predict, labels).float().sum().item()
                        total_samples_s += batch_size
                    acc_s = (total_correct_s / total_samples_s) * 100

                # ======== Target-domain evaluation ========
                combined_dataset = ConcatDataset(target_test_loader)
                dataloader = DataLoader(combined_dataset, batch_size=len(combined_dataset))
                total_correct_t, total_samples_t = 0, 0
                with torch.no_grad():
                    for batch_x_0, batch_y_0, batch_domain_0 in dataloader:
                        inputs = batch_x_0.to(self.device)
                        labels = batch_y_0.to(self.device)
                        logits = self.model.model_inference(inputs)
                        predict = logits.argmax(dim=1)
                        total_correct_t += torch.eq(predict, labels).float().sum().item()
                        total_samples_t += labels.size(0)
                    acc_t = (total_correct_t / total_samples_t) * 100

                    logging.info("Soure_domain_acc: {:.2f}, Target_domain_acc: {:.2f}".format(acc_s, acc_t))

                    # ======== Save the best model (based on target-domain accuracy) ========
                    if acc_t >= best_acc:
                        best_acc = acc_t
                        save_dir = os.path.join('./trained_models', self.args.dataset_name, self.args.model_name, s_domain)
                        os.makedirs(save_dir, exist_ok=True)
                        torch.save(self.model.state_dict(), os.path.join(save_dir, f'operation_{op_num}.pth'))

            # ======== Calculate average time per epoch ========
            average_time = sum(time_list) / len(time_list)
            logging.info("Average time per epoch: {:.2f} ms".format(average_time * 1e3))

        return

    def test(self, op_num, source_test_loader, target_test_loader, s_domain):
        """
        Test function: evaluate the best model on both the target domain and the source domain.
        """

        # ====== Load the pre-trained best model weights ======
        save_dir = os.path.join(f'./trained_models/{self.args.dataset_name}/{self.args.model_name}/{s_domain}')
        model_path = os.path.join(save_dir, f'operation_{op_num}.pth')
        self.model.load_state_dict(torch.load(model_path), strict=False)

        # ====== Target-domain testing ======
        test_start_time_t = time.time()
        total_loss, total_samples = 0, 0
        all_preds, all_labels = [], []
        combined_dataset = ConcatDataset(target_test_loader)
        dataloader = DataLoader(combined_dataset, batch_size=len(combined_dataset), shuffle=False)
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, labels, domain) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                logits = self.model.model_inference(inputs)
                loss = self.criterion(logits, labels)

                total_samples += labels.size(0)
                predictions = logits.argmax(dim=1)
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                total_loss += loss.item()

        # Calculate source-domain metrics: average loss, accuracy, and  F1 score
        avg_loss = total_loss / total_samples
        avg_acc_t = accuracy_score(all_labels, all_preds) * 100
        avg_f1_t = f1_score(all_labels, all_preds, average='weighted') * 100
        test_duration = time.time() - test_start_time_t
        logging.info(
            f'Operation_{op_num}, Target test Loss: {avg_loss:.4f}, Acc: {avg_acc_t:.4f}, '
            f'F1: {avg_f1_t:.4f}, Time: {test_duration:.4f} sec'
        )

        # ====== Source-domain testing ======
        test_start_time_s = time.time()
        all_preds, all_labels = [], []
        total_loss, total_samples = 0, 0
        with torch.no_grad():
            self.model.eval()
            for batch_idx, (inputs, labels, domain) in enumerate(source_test_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                logits = self.model.model_inference(inputs)
                loss = self.criterion(logits, labels)


                total_loss += loss.item()
                total_samples += labels.size(0)
                predictions = logits.argmax(dim=1)
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # Calculate target-domain metrics: average loss, accuracy, and  F1 score
            avg_loss = total_loss / total_samples
            avg_ac_s = accuracy_score(all_labels, all_preds) * 100
            epoch_f1_s = f1_score(all_labels, all_preds, average='weighted') * 100
            test_duration = time.time() - test_start_time_s
            logging.info(
                f'Operation_{op_num}, Source test Loss: {avg_loss:.4f}, Acc: {avg_ac_s:.4f}, '
                f'F1: {epoch_f1_s:.4f}, Time: {test_duration:.4f} sec')
        return avg_acc_t, avg_f1_t




