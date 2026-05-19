import os
import sys

__package__ = "trainer"  # 告诉Python当前文件属于哪个包，便于相对导入
# __package__是一个内置变量，与__name__类似，Python天生就给每个文件准备的内置变量，系统自带

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # 总的功能就是把上级目录加入模块搜索路径
# __file__内置变量，当前这个.py文件的完整路径
# os.path.dirname(__file__)返回路径名的目录部分，也就是取当前文件所在文件夹路径
# .join()拼接这两个路径,..表示返回上一级目录
# abspath()把相对路径转换为绝对路径
# sys.path.append() 把一个路径添加到Python的模块搜索路径列表里

# 实际上就是sys.path.append("D:\\code_file\\github_file\\MiniMind\\minimind_git\\minimind")

# 便于后面导入model.model_minimind和dataset.lm_dataset

import argparse  # 解析命令行参数
import time
import warnings
import torch
import torch.distributed as dist  # 多卡分布式训练
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig  # 模型配置
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import (
    get_lr,
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    init_model,
    SkipBatchSampler,
)

warnings.filterwarnings("ignore")  # 忽略所有警告


# 训练的一个epoch循环
# epoch表示现在是当前的第几个epoch
# iters这个epoch总共有多少step
# start_stop断点续训时从哪个step开始
# train_epoch使用了外部全局变量
def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    # wandb日志记录器
    start_time = time.time()
    last_step = start_step
    # 遍历DataLoader,enumerate会同时给出step编号和batch数据，每次给出一个batch数据，也就是iters或者说step
    # start_step=0，那么step从1开始
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)  # 输入的tokenid 以及标签搬到GPU
        labels = labels.to(args.device)
        last_step = step
        # 学习率衰减策略，每个step都重新计算学习率,get_lr(current_step, total_steps, lr)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)

        # optimizer.param_groups是优化器内部的参数组，一个优化器可以有多个参数组，每组可以有不同学习率
        for param_group in optimizer.param_groups:
            # 把当前学习率写到优化器中
            param_group["lr"] = lr

        # 上下文管理器，表示在这个代码块内部Pytorch会自动判断哪些操作适合用低精度计算
        with autocast_ctx:
            # 自动计算next-token预测的损失
            # MiniMindForCausalLM.forward(...)在labels不为空时，把logits[...,:-1,:]和labels[...,1:]对齐
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss

            # 梯度累积，每个小batch的loss都除以args.accumulation_steps
            loss = loss / args.accumulation_steps

        # GradScaler梯度缩放，先把loss放大很多倍，反向传播时的梯度跟着放大
        # 这里只是反向传播，但是不更新参数，不清空梯度
        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:

            # 将优化器管理的所有参数的梯度除以缩放因子（还原真实梯度）
            scaler.unscale_(optimizer)
            # 梯度裁剪，args.grad_clip梯度裁剪的阈值
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 这才是真正的更新参数，普通写法是optimizer.step()，混合精度用下面的方法
            scaler.step(optimizer)
            # GradScaler会动态调整缩放倍数，这里是根据本次训练情况更新下一次使用的scale倍数
            scaler.update()

            # 清空梯度
            optimizer.zero_grad(set_to_none=True)

        # 打印日志
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time

            # loss.item()将零维张量转为Python浮点数，乘回acc_steps,恢复为原始平均损失
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]["lr"]
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min"
            )
            if wandb:
                wandb.log(
                    {
                        "loss": current_loss,
                        "logits_loss": current_logits_loss,
                        "aux_loss": current_aux_loss,
                        "learning_rate": current_lr,
                        "epoch_time": eta_min,
                    }
                )

        # 保存模型
        # is_main_process()，分布式训练时，只有主进程执行这部分内容
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"

            raw_model = (
                model.module if isinstance(model, DistributedDataParallel) else model
            )
            # 如果使用了torch.compile，编译后的模型有一个_orig_mod属性指向编译前的模型，这里保证我们保存的时未编译的原始状态
            raw_model = getattr(raw_model, "_orig_mod", raw_model)

            # raw_model.state_dict()获取模型参数字典，键是参数名，值是张量
            state_dict = raw_model.state_dict()

            # 字典推导式，将所有参数转为半精度，并移动到GPU，减少存储空间
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir="../checkpoints",
            )
            model.train()
            del state_dict

        del input_ids, labels, res, loss

    # 处理不完整的累积步数
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    # 命令行参数解析
    # 允许我们在命令行里覆盖默认参数  例如 python train_pretrain.py --batch_size 4
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")

    # 在trainer/目录下运行，所以../out就是和trainer同级的文件夹  minimind/out
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument(
        "--save_weight", default="pretrain", type=str, help="保存权重的前缀名"
    )
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="训练设备",
    )
    # 混合精度类型
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")

    # DataLoader 的数据加载进程数
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")

    # 梯度累积步数，也就是不是每个batch就更新一次参数，而是多个batch更新一次参数
    parser.add_argument(
        "--accumulation_steps", type=int, default=8, help="梯度累积步数"
    )
    # 防止梯度突然变得特别大
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument("--hidden_size", default=768, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层数量")
    parser.add_argument(
        "--max_seq_len",
        default=340,
        type=int,
        help="训练的最大截断长度（中文1token≈1.5~1.7字符）",
    )
    parser.add_argument(
        "--use_moe",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用MoE架构（0=否，1=是）",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../dataset/pretrain_t2t_mini.jsonl",
        help="预训练数据路径",
    )
    # 是否基于已有权重继续训练，默认none，从随机初始化开始训练
    parser.add_argument(
        "--from_weight",
        default="none",
        type=str,
        help="基于哪个权重训练，为none则从头开始",
    )
    # 断点续训练，0不恢复，1自动读取checkpoint恢复，例如python train_pretrain.py --from_resume 1
    parser.add_argument(
        "--from_resume",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否自动检测&续训（0=否，1=是）",
    )
    # action="store_true"表示python train.py --use_wandb   → 开启 wandb weight and Biases
    # 云端的实时仪表盘，自动画曲线
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument(
        "--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名"
    )
    # torch.compile是Pytorch2.0+的加速功能
    parser.add_argument(
        "--use_compile",
        default=0,
        type=int,
        choices=[0, 1],
        help="是否使用torch.compile加速（0=否，1=是）",
    )
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(
        args.save_dir, exist_ok=True
    )  # 递归创建目录，exist_ok=True,就是目录可以已经存在，不会报错，反之
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )
    # 如果--from_resume 1，就会尝试读取../checkpoints里的断点文件
    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir="../checkpoints")
        if args.from_resume == 1
        else None
    )

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # 如果是cuda启用自动混合精度训练
    autocast_ctx = (
        nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    )

    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(
            project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume
        )

    # ========== 5. 定义模型、数据、优化器 ==========
    # 加载 tokenizer，创建 MiniMindForCausalLM 模型
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # 把jsonl数据变成Pytorch Dataset
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

    # DistributedSampler分布式训练时，用于将数据集的不同部分分配给不同GPU，避免数据重复
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # GradScaler给float16使用的梯度缩放器
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

    # AdamW优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0

    # 如果是断点续训，恢复模型参数，优化器状态，GradScaler状态，训练到第几个epoch,训练到第几个step
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    # ========== 7. 编译和分布式包装 ==========
    # torch.compile是Pytorch2.x的编译加速功能
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger("torch.compile enabled")

    # 多卡训练的包装
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):

        # Python的短路写法
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)

        #  torch.randperm(n)生成 0~n-1 随机打乱，不重复，全覆盖
        # 打乱后的Python列表
        indices = torch.randperm(len(train_ds)).tolist()

        # 断点续训服务
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0

        # SkipBatchSampler自定义的采样器类，能够跳过前 skip 个 batch
        batch_sampler = SkipBatchSampler(
            train_sampler or indices, args.batch_size, skip
        )
        # batch_sampler 和 batch_size 不能同时指定；这里用 batch_sampler 控制批次顺序
        # batch_sampler 已经完整定义了“如何将数据集索引分组”
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        if skip > 0:
            Logger(
                f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始"
            )

            # train_epoch(epoch, loader, iters, start_step=0, wandb=None)
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.destroy_process_group()
