import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ====================== 配置类 ======================
class Config:
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据参数
    seq_len = 24 * 7  # 一周数据 (小时级)
    input_dim = 12  # 特征维度: wind, solar, electric, cooling, heating, temperature, day_cos, day_sin, hour_cos, hour_sin, mean, std
    noise_dim = 32  # 噪声维度
    latent_dim = 64  # 潜在空间维度
    cond_dim = 4  # 季节条件向量维度
    
    # 模型架构
    hidden_dim = 128
    num_layers = 3
    dropout = 0.1
    
    # 训练参数
    batch_size = 256
    epochs_embed = 100
    epochs_supervise = 200
    epochs_joint = 300
    lr = 0.001
    gamma = 0.1  # 判别器损失系数
    
    # 优化器
    use_amp = True  # 自动混合精度
    lr_scheduler = True
    
    # 路径设置
    data_paths = {
        "spring": "spring_data.csv",
        "summer": "summer_data.csv",
        "autumn": "autumn_data.csv",
        "winter": "winter_data.csv"
    }
    output_dir = "./timegan_results/"
    
    def __init__(self):
        os.makedirs(self.output_dir, exist_ok=True)

# ====================== 模型组件 ======================
class Embedder(nn.Module):
    """ 将原始数据映射到潜在空间 """
    def __init__(self, config):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_dim, config.latent_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        # CUDA断言: 输入维度检查
        assert x.dim() == 3, f"Expected 3D tensor (batch, seq, features), got {x.dim()}D"
        output, _ = self.rnn(x)
        return self.fc(output[:, -1, :])  # 只取最后时间步

class Recovery(nn.Module):
    """ 将潜在表示映射回原始空间 """
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU()
        )
        self.rnn = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.input_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout
        )
    
    def forward(self, h):
        h_expanded = self.fc(h).unsqueeze(1).repeat(1, Config.seq_len, 1)
        output, _ = self.rnn(h_expanded)
        return output

class BayesianGenerator(nn.Module):
    """ 贝叶斯生成器: 从噪声生成潜在表示 """
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 条件向量处理层
        self.cond_fc = nn.Linear(config.cond_dim, config.noise_dim)
        
        # RNN主干网络
        self.rnn = nn.GRU(
            input_size=config.noise_dim * 2,  # 噪声 + 条件
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout
        )
        
        # 输出层 (均值和log方差)
        self.fc_mu = nn.Linear(config.hidden_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(config.hidden_dim, config.latent_dim)
    
    def forward(self, z, cond):
        # CUDA断言: 维度检查
        assert z.dim() == 3, f"z must be 3D tensor, got {z.dim()}D"
        assert cond.dim() == 2, f"cond must be 2D tensor, got {cond.dim()}D"
        
        # 条件向量处理
        cond_proj = self.cond_fc(cond)
        cond_expanded = cond_proj.unsqueeze(1).repeat(1, z.size(1), 1)
        
        # 拼接噪声和条件
        z_cond = torch.cat([z, cond_expanded], dim=-1)
        
        # RNN处理
        rnn_out, _ = self.rnn(z_cond)
        
        # 计算均值和方差
        mu = self.fc_mu(rnn_out)
        logvar = self.fc_logvar(rnn_out)
        
        # 重参数采样
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        x_hat = mu + eps * std
        
        return x_hat, mu, logvar

class Discriminator(nn.Module):
    """ 判别器: 区分真实/生成的潜在表示 """
    def __init__(self, config):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=config.latent_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, h):
        output, _ = self.rnn(h)
        return self.fc(output[:, -1, :])  # 取最后时间步输出

class Supervisor(nn.Module):
    """ 监督器: 学习时间序列的动态特性 """
    def __init__(self, config):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=config.latent_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers-1,  # 比生成器少一层
            batch_first=True,
            dropout=config.dropout
        )
        self.fc = nn.Linear(config.hidden_dim, config.latent_dim)
    
    def forward(self, h):
        output, _ = self.rnn(h)
        return self.fc(output)

# ====================== 完整TimeGAN模型 ======================
class TimeGAN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 模块初始化
        self.embedder = Embedder(config).to(config.device)
        self.recovery = Recovery(config).to(config.device)
        self.generator = BayesianGenerator(config).to(config.device)
        self.discriminator = Discriminator(config).to(config.device)
        self.supervisor = Supervisor(config).to(config.device)
        
        # 优化器
        self.opt_embed = optim.Adam(
            list(self.embedder.parameters()) + list(self.recovery.parameters()),
            lr=config.lr
        )
        self.opt_supervise = optim.Adam(
            self.supervisor.parameters(),
            lr=config.lr
        )
        self.opt_joint = optim.Adam(
            list(self.generator.parameters()) + 
            list(self.discriminator.parameters()) +
            list(self.embedder.parameters()) +
            list(self.recovery.parameters()) +
            list(self.supervisor.parameters()),
            lr=config.lr
        )
        
        # 学习率调度器
        if config.lr_scheduler:
            self.scheduler_embed = optim.lr_scheduler.StepLR(self.opt_embed, step_size=50, gamma=0.5)
            self.scheduler_supervise = optim.lr_scheduler.StepLR(self.opt_supervise, step_size=50, gamma=0.5)
            self.scheduler_joint = optim.lr_scheduler.StepLR(self.opt_joint, step_size=100, gamma=0.5)
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
        # 混合精度训练
        self.scaler = GradScaler(enabled=config.use_amp)
        
    def compute_kl_loss(self, mu, logvar):
        """ 计算KL散度损失 """
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (mu.size(0) * mu.size(1))
    
    def compute_moment_loss(self, real, fake):
        """ 统计矩匹配损失 """
        real_mean, real_std = real.mean(), real.std()
        fake_mean, fake_std = fake.mean(), fake.std()
        loss_v1 = self.mse_loss(real_mean, fake_mean)
        loss_v2 = self.mse_loss(real_std, fake_std)
        return loss_v1 + loss_v2
    
    def forward(self, x, cond, mode="train"):
        # 嵌入阶段
        h = self.embedder(x)
        x_tilde = self.recovery(h)
        
        # 监督阶段
        h_hat_supervise = self.supervisor(h[:, :-1, :])
        
        # 生成阶段
        z = torch.randn(x.size(0), self.config.seq_len, self.config.noise_dim).to(self.config.device)
        e_hat, mu, logvar = self.generator(z, cond)
        
        # 判别阶段
        y_real = self.discriminator(h)
        y_fake = self.discriminator(e_hat)
        y_fake_e = self.discriminator(e_hat.detach())
        
        return {
            "x_tilde": x_tilde,
            "h": h,
            "h_hat_supervise": h_hat_supervise,
            "e_hat": e_hat,
            "y_real": y_real,
            "y_fake": y_fake,
            "y_fake_e": y_fake_e,
            "mu": mu,
            "logvar": logvar
        }

# ====================== 数据处理工具 ======================
class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        self.cond_vectors = {
            "spring": [1, 0, 0, 0],
            "summer": [0, 1, 0, 0],
            "autumn": [0, 0, 1, 0],
            "winter": [0, 0, 0, 1]
        }
    
    def load_and_process(self):
        all_data = []
        all_conditions = []
        
        for season, path in self.config.data_paths.items():
            # 加载数据
            df = pd.read_csv(path)
            data = df.values
            
            # 归一化
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            self.scalers[season] = scaler
            
            # 创建序列
            num_samples = scaled_data.shape[0] - self.config.seq_len + 1
            sequences = np.zeros((num_samples, self.config.seq_len, self.config.input_dim))
            
            for i in range(num_samples):
                sequences[i] = scaled_data[i:i+self.config.seq_len]
            
            # 创建条件向量
            conditions = np.tile(self.cond_vectors[season], (num_samples, 1))
            
            all_data.append(sequences)
            all_conditions.append(conditions)
        
        # 合并所有季节数据
        X = np.concatenate(all_data, axis=0)
        C = np.concatenate(all_conditions, axis=0)
        
        return torch.tensor(X, dtype=torch.float32), torch.tensor(C, dtype=torch.float32)
    
    def inverse_transform(self, data, season):
        """ 反归一化数据 """
        scaler = self.scalers[season]
        return scaler.inverse_transform(data)

# ====================== 训练循环 ======================
def train_embedding(model, dataloader, config):
    """ 嵌入训练阶段 """
    model.train()
    losses = []
    
    for epoch in range(config.epochs_embed):
        epoch_loss = 0.0
        
        for x, c in dataloader:
            x = x.to(config.device)
            
            with autocast(enabled=config.use_amp):
                # 前向传播
                h = model.embedder(x)
                x_tilde = model.recovery(h)
                
                # 计算损失
                loss = model.mse_loss(x_tilde, x)
            
            # 反向传播
            model.opt_embed.zero_grad()
            model.scaler.scale(loss).backward()
            model.scaler.step(model.opt_embed)
            model.scaler.update()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if config.lr_scheduler:
            model.scheduler_embed.step()
        
        print(f"Embedding Epoch [{epoch+1}/{config.epochs_embed}] Loss: {avg_loss:.6f}")
    
    return losses

def train_supervisor(model, dataloader, config):
    """ 监督训练阶段 """
    model.train()
    losses = []
    
    for epoch in range(config.epochs_supervise):
        epoch_loss = 0.0
        
        for x, c in dataloader:
            x = x.to(config.device)
            
            with autocast(enabled=config.use_amp):
                # 前向传播
                h = model.embedder(x)
                h_hat_supervise = model.supervisor(h[:, :-1, :])
                
                # 监督损失
                loss = model.mse_loss(h_hat_supervise, h[:, 1:, :])
            
            # 反向传播
            model.opt_supervise.zero_grad()
            model.scaler.scale(loss).backward()
            model.scaler.step(model.opt_supervise)
            model.scaler.update()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if config.lr_scheduler:
            model.scheduler_supervise.step()
        
        print(f"Supervisor Epoch [{epoch+1}/{config.epochs_supervise}] Loss: {avg_loss:.6f}")
    
    return losses

def train_joint(model, dataloader, config):
    """ 联合训练阶段 """
    model.train()
    history = {
        "G_loss": [], "D_loss": [], "E_loss": [],
        "G_adv": [], "G_supervise": [], "G_moment": [], "G_kl": []
    }
    
    for epoch in range(config.epochs_joint):
        epoch_losses = {k: 0.0 for k in history.keys()}
        
        for x, c in dataloader:
            x, c = x.to(config.device), c.to(config.device)
            
            # ===== 判别器训练 =====
            with autocast(enabled=config.use_amp):
                # 前向传播
                outputs = model(x, c)
                
                # 判别器损失
                D_loss_real = model.bce_loss(outputs["y_real"], torch.ones_like(outputs["y_real"]))
                D_loss_fake = model.bce_loss(outputs["y_fake"], torch.zeros_like(outputs["y_fake"]))
                D_loss_fake_e = model.bce_loss(outputs["y_fake_e"], torch.zeros_like(outputs["y_fake_e"]))
                D_loss = D_loss_real + D_loss_fake + config.gamma * D_loss_fake_e
            
            model.opt_joint.zero_grad()
            model.scaler.scale(D_loss).backward(retain_graph=True)
            model.scaler.step(model.opt_joint)
            
            # ===== 生成器训练 =====
            with autocast(enabled=config.use_amp):
                # 对抗损失
                G_loss_U = model.bce_loss(outputs["y_fake"], torch.ones_like(outputs["y_fake"]))
                G_loss_U_e = model.bce_loss(outputs["y_fake_e"], torch.ones_like(outputs["y_fake_e"]))
                
                # 监督损失
                G_loss_S = model.mse_loss(outputs["h"][:, 1:, :], outputs["h_hat_supervise"][:, :-1, :])
                
                # 统计矩匹配损失
                G_loss_V = model.compute_moment_loss(outputs["h"], outputs["e_hat"])
                
                # KL散度损失
                G_KL_loss = model.compute_kl_loss(outputs["mu"], outputs["logvar"])
                
                # 嵌入和恢复损失
                E_loss_T0 = model.mse_loss(outputs["x_tilde"], x)
                E_loss0 = model.mse_loss(outputs["h"], model.embedder(x))
                E_loss = E_loss0 + 0.1 * G_loss_S
                
                # 总损失
                G_loss = G_loss_U + G_loss_U_e + 100 * torch.sqrt(G_loss_S) + 100 * G_loss_V + G_KL_loss + E_loss
            
            model.opt_joint.zero_grad()
            model.scaler.scale(G_loss).backward()
            model.scaler.step(model.opt_joint)
            model.scaler.update()
            
            # 记录损失
            losses = {
                "G_loss": G_loss.item(),
                "D_loss": D_loss.item(),
                "E_loss": E_loss.item(),
                "G_adv": (G_loss_U + G_loss_U_e).item(),
                "G_supervise": G_loss_S.item(),
                "G_moment": G_loss_V.item(),
                "G_kl": G_KL_loss.item()
            }
            
            for k in epoch_losses:
                epoch_losses[k] += losses[k]
        
        # 计算平均损失
        for k in epoch_losses:
            epoch_losses[k] /= len(dataloader)
            history[k].append(epoch_losses[k])
        
        if config.lr_scheduler:
            model.scheduler_joint.step()
        
        print(f"Joint Epoch [{epoch+1}/{config.epochs_joint}] G_loss: {epoch_losses['G_loss']:.6f} D_loss: {epoch_losses['D_loss']:.6f}")
    
    return history

# ====================== 可视化工具 ======================
def plot_losses(embed_loss, super_loss, joint_history, config):
    """ 绘制损失曲线 """
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.figsize': (15, 10),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'lines.linewidth': 2.5,
        'legend.fontsize': 14
    })
    colors = sns.color_palette("husl", 8)
    
    # 嵌入损失
    plt.figure()
    plt.plot(embed_loss, color=colors[0])
    plt.title("Embedding Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, "embed_loss.png"))
    
    # 监督损失
    plt.figure()
    plt.plot(super_loss, color=colors[1])
    plt.title("Supervisor Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, "supervise_loss.png"))
    
    # 联合训练损失
    plt.figure(figsize=(15, 10))
    
    # 总损失
    plt.subplot(2, 2, 1)
    plt.plot(joint_history["G_loss"], label="Generator Loss", color=colors[0])
    plt.plot(joint_history["D_loss"], label="Discriminator Loss", color=colors[1])
    plt.plot(joint_history["E_loss"], label="Embedding Loss", color=colors[2])
    plt.title("Total Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 生成器损失分解
    plt.subplot(2, 2, 2)
    plt.plot(joint_history["G_adv"], label="Adversarial Loss", color=colors[0])
    plt.plot(joint_history["G_supervise"], label="Supervise Loss", color=colors[1])
    plt.plot(joint_history["G_moment"], label="Moment Loss", color=colors[2])
    plt.plot(joint_history["G_kl"], label="KL Loss", color=colors[3])
    plt.title("Generator Loss Components")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 对抗损失细节
    plt.subplot(2, 2, 3)
    plt.plot(joint_history["G_adv"], label="Generator Adv Loss", color=colors[0])
    plt.plot(joint_history["D_loss"], label="Discriminator Loss", color=colors[1])
    plt.title("Adversarial Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # KL和矩损失
    plt.subplot(2, 2, 4)
    plt.plot(joint_history["G_kl"], label="KL Divergence", color=colors[3])
    plt.plot(joint_history["G_moment"], label="Moment Matching", color=colors[2])
    plt.title("Bayesian and Statistical Losses")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, "joint_training_losses.png"))

# ====================== 主函数 ======================
def main():
    # 初始化配置
    config = Config()
    print(f"Using device: {config.device}")
    
    # 数据处理
    processor = DataProcessor(config)
    X, C = processor.load_and_process()
    dataset = TensorDataset(X, C)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # 初始化模型
    model = TimeGAN(config)
    
    # 训练阶段
    print("Starting Embedding Training...")
    embed_loss = train_embedding(model, dataloader, config)
    
    print("\nStarting Supervisor Training...")
    super_loss = train_supervisor(model, dataloader, config)
    
    print("\nStarting Joint Training...")
    joint_history = train_joint(model, dataloader, config)
    
    # 可视化损失
    plot_losses(embed_loss, super_loss, joint_history, config)
    
    # 保存模型
    torch.save({
        'embedder': model.embedder.state_dict(),
        'recovery': model.recovery.state_dict(),
        'generator': model.generator.state_dict(),
        'discriminator': model.discriminator.state_dict(),
        'supervisor': model.supervisor.state_dict(),
        'config': config
    }, os.path.join(config.output_dir, "timegan_model.pth"))
    
    print("Training completed and model saved!")

if __name__ == "__main__":
    # CUDA断言错误检查
    torch.autograd.set_detect_anomaly(True)
    main()