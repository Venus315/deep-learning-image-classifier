import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datetime import datetime
from tqdm import tqdm
import logging

from core import model_custom_vgg

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class TrainingManager:
    def __init__(self):
        self.model = model_custom_vgg().to(DEVICE)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = self._setup_scheduler()

        self.output_dir = os.path.join("checkpoints", datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.output_dir, exist_ok=True)

        self.store_epochs = {29, 49, 99, 149}
        self.top_acc = 0.0
        self.batch_size = 128

        self.loader_train, self.loader_test = self._load_data()

    def _load_data(self):
        aug_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        norm_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4942, 0.4851, 0.4504), (0.2467, 0.2429, 0.2616))
        ])

        train_ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=aug_train)
        test_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=norm_test)

        return (
            DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=2),
            DataLoader(test_ds, batch_size=100, shuffle=False, num_workers=2)
        )

    def _setup_scheduler(self):
        return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

    def train_loop(self, epoch):
        self.model.train()
        total_seen, total_hit, total_loss = 0, 0, 0.0

        for xb, yb in tqdm(self.loader_train, desc=f"[Epoch {epoch}] Training..."):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = self.model(xb)
            loss_val = self.loss_fn(pred, yb)

            self.optimizer.zero_grad()
            loss_val.backward()
            self.optimizer.step()

            total_seen += yb.size(0)
            total_hit += pred.argmax(1).eq(yb).sum().item()
            total_loss += loss_val.item() * xb.size(0)

        acc = 100.0 * total_hit / total_seen
        logger.info(f"✅ Epoch {epoch} | Train Accuracy: {acc:.2f}% | Loss: {total_loss / total_seen:.4f}")

    def evaluate_model(self):
        self.model.eval()
        total_correct, eval_loss, count = 0, 0.0, 0

        with torch.no_grad():
            for xb, yb in self.loader_test:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = self.model(xb)
                loss = self.loss_fn(pred, yb)

                eval_loss += loss.item() * xb.size(0)
                total_correct += pred.argmax(1).eq(yb).sum().item()
                count += yb.size(0)

        return 100.0 * total_correct / count, eval_loss / count

    def persist_model(self, metric, epoch):
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'score': metric
        }
        torch.save(state, os.path.join(self.output_dir, f"snapshot_epoch{epoch+1}.pth"))
        logger.info(f"📁 Saved snapshot for epoch {epoch}.")

        if metric > self.top_acc:
            self.top_acc = metric
            torch.save(state, os.path.join(self.output_dir, "best_model.pth"))
            logger.info("🏅 New best model stored.")

    def execute_all(self, max_epochs=200):
        for ep in range(max_epochs):
            self.train_loop(ep)
            val_score, val_loss = self.evaluate_model()
            logger.info(f"📊 Epoch {ep} | Eval Acc: {val_score:.2f}% | Eval Loss: {val_loss:.4f}")

            if ep in self.store_epochs or val_score > self.top_acc:
                self.persist_model(val_score, ep)
            self.scheduler.step()


def run_experiment():
    task = TrainingManager()
    task.execute_all()


if __name__ == "__main__":
    run_experiment()
