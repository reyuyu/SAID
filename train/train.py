import torch
# from utils import concat_all_gather, is_dist_avail_and_initialized, accuracy
# the original concat_all_gather is abandoned because of no gradient backward
from train_utils import is_dist_avail_and_initialized, accuracy
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")

from sharegpt4v import share4v_val_dataset, share4v_train_dataset
from model import longclip
from torch.utils.data.distributed import DistributedSampler
from scheduler import cosine_lr
import argparse
import subprocess
import collections
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
from torch.cuda.amp import GradScaler
from train_utils import LossManager, prepare_sub_directories, get_run_id, eval_coco

# --------------------------------------------------------------------------- #
# SmartCLIP reproduction support (Phase 3.0A baseline validation)
#
# These helpers are opt-in: with no ``--output_dir`` the trainer behaves exactly as the
# original (``./runs/<auto id>``) and, with no ``--seed``, the RNG state is left untouched.
# Nothing here changes the loss, the mask, the optimizer grouping, the scheduler, the model
# forward or the dataset caption sampling.
# --------------------------------------------------------------------------- #
def seed_everything(seed):
	"""Fix python / numpy / torch RNGs. Called only when ``--seed`` is given."""
	import random
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def state_digest(state_dict):
	"""Deterministic SHA-256 over a state dict (weights only, no RNG consumed)."""
	import hashlib
	digest = hashlib.sha256()
	for key in sorted(state_dict):
		tensor = state_dict[key]
		digest.update(key.encode('utf-8'))
		if torch.is_tensor(tensor):
			digest.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
		else:
			digest.update(repr(tensor).encode('utf-8'))
	return digest.hexdigest()


class CLIP_Clean_Train():
	def __init__(self, rank, local_rank, args):
		self.rank = rank
		self.local_rank = local_rank
		self.base_model = args.base_model
		self.model, self.preprocess = longclip.load_from_clip(self.base_model, device='cpu',
		                                                      download_root=args.download_root, args=args)
		self.model.train()
		self.model.logit_scale = torch.nn.Parameter(torch.ones([]) * args.log_scale)
		self.model = self.model.cuda()
		if getattr(args, 'init_state', None):
			# opt-in: load a frozen initial state dict so a comparison run starts from exactly
			# the same weights as the reference models. The `clip.`-prefixed SALU layout and a
			# bare CLIP layout are both accepted; only mask_net may be absent (a bare CLIP
			# checkpoint has none), and anything else must match or we fail loudly.
			state = torch.load(args.init_state, map_location='cpu', weights_only=False)
			if isinstance(state, dict) and 'model' in state:
				state = state['model']
			target = set(self.model.state_dict())
			# accept the SALU layout (``clip.*`` + ``said_router.*``) as well as a bare CLIP
			# layout: map each source key onto a target key by name, with or without the
			# evaluator's ``clip.`` prefix
			mapped = {}
			for key, value in state.items():
				if key in target:
					mapped[key] = value
				elif key.startswith('clip.') and key[len('clip.'):] in target:
					mapped[key[len('clip.'):]] = value
				elif ('clip.' + key) in target:
					mapped['clip.' + key] = value
			if not mapped:
				raise ValueError('--init_state %s matches no model parameter' % args.init_state)
			missing, unexpected = self.model.load_state_dict(mapped, strict=False)
			bad_missing = [k for k in missing if not k.startswith('mask_net')]
			if bad_missing or list(unexpected):
				raise RuntimeError('--init_state did not load cleanly: missing %r unexpected %r'
				                   % (bad_missing, list(unexpected)))
			print('INIT_STATE_LOADED %s tensors=%d mask_net_absent=%d'
			      % (args.init_state, len(mapped), len(missing)), flush=True)

		self.batch_size = args.batch_size
		self.num_epoch = args.epochs
		self.lr = args.lr
		self.weight_decay = args.weight_decay
		self.warmup_length = args.warmup_length
		self.args = args
		run_id = get_run_id('runs')
		target_dir = './runs/' + '%06d_%s_totalbs1024_sparse%s_align%s_mask%s_%s' % (
		run_id, args.base_model[-2:], args.lambda_sparse, args.lambda_align,
		args.mask_lr, 'soft' if args.soft_mask else 'hard'
		)
		if getattr(args, 'output_dir', None):
			# opt-in: a caller-chosen directory instead of the auto-numbered ./runs/<id>
			target_dir = args.output_dir
		self.logdir = target_dir
		if self.rank == 0:
			os.makedirs(target_dir, exist_ok=True)
			#prepare_sub_directories(target_dir)
			self.loss_manager = LossManager(file_path=os.path.join(target_dir, "loss.txt"))
			self.metric_manager = LossManager(file_path=os.path.join(target_dir, "metric.txt"))
			print(args)
			if getattr(args, 'output_dir', None):
				# reproducibility record: initial-state digest + the effective configuration.
				# Everything here is computed locally: it must NOT construct another dataset
				# instance (that would consume the RNG stream the training data uses) and it
				# must not depend on attributes the constructor sets later.
				import json as _json
				_world = torch.distributed.get_world_size()
				_accum = 1024 // args.batch_size // _world
				record = {
					'run': 'smartclip-reproduction',
					'base_model': args.base_model,
					'initial_state_sha256': state_digest(self.model.state_dict()),
					'world_size': _world,
					'batch_size_per_gpu': args.batch_size,
					'accumulation_steps': _accum,
					'effective_batch_size': args.batch_size * _accum * _world,
					'epochs': args.epochs,
					'lr': args.lr,
					'mask_lr': args.mask_lr,
					'weight_decay': args.weight_decay,
					'lambda_sparse': args.lambda_sparse,
					'lambda_align': args.lambda_align,
					'soft_mask': args.soft_mask,
					'warmup_length': args.warmup_length,
					'seed': getattr(args, 'seed', None),
					'log_scale': args.log_scale,
					'max_steps': args.max_steps,
					'dataset_json': os.environ.get('SHARE4V_JSON'),
					'dataset_json_size_bytes': (os.path.getsize(os.path.join(
						os.environ.get('SHARE4V_DATA_ROOT', '../datasets/ShareGPT4V'),
						os.environ.get('SHARE4V_JSON', '')))
						if os.environ.get('SHARE4V_JSON') else None),
				}
				with open(os.path.join(target_dir, 'reproducibility.json'), 'w') as handle:
					_json.dump(record, handle, indent=2, sort_keys=True)
				print('REPRO_INITIAL_STATE_SHA256', record['initial_state_sha256'], flush=True)
				print('EFFECTIVE_BATCH_SIZE', record['effective_batch_size'],
				      'ACCUMULATION_STEPS', record['accumulation_steps'], flush=True)
		self.target_dir = target_dir

		self.writer = SummaryWriter(self.logdir)

		self.accumulation_steps = 1024 // self.batch_size // torch.distributed.get_world_size()  # Set the number of accumulation steps
		self.effective_batch_size = self.batch_size * self.accumulation_steps * torch.distributed.get_world_size()
		print('>>>>> effective_batch_size <<<<<<', self.effective_batch_size, ' >>> accmulate steps ',
		      self.accumulation_steps)

		self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank],
		                                                       find_unused_parameters=True)
		self.model._set_static_graph()

		mask_net_params = []
		other_params = []

		for name, param in self.model.named_parameters():
			if 'mask_net' in name:
				mask_net_params.append(param)
			else:
				other_params.append(param)

		self.mask_lr = args.mask_lr
		self.optimizer = optim.AdamW(other_params, lr=self.lr, weight_decay=self.weight_decay)
		self.mask_net_optimizer = optim.AdamW(mask_net_params, lr=self.mask_lr, weight_decay=0)
		self.scaler = GradScaler()


	def train_epoch(self, dataloader, epoch, start_iter=0, args=None):
		running_loss = 0.0
		running_loss_short = 0.0
		# rank = torch.distributed.get_rank()
		num_batches_per_epoch = len(dataloader)
		print('Start Training ', epoch)
		# print('>>>>> num_batches_per_epoch <<<<<<', num_batches_per_epoch)
		self.model.train()
		if self.rank == 0:
			pbar = tqdm(total=num_batches_per_epoch)
		for i, (images, texts) in enumerate(dataloader):
			# print(images.size(), '>>>>>>> train iter')
			step = num_batches_per_epoch * epoch + i
			if step < start_iter:
				continue
			if self.args.max_steps is not None and step >= self.args.max_steps:
				self.max_steps_reached = True
				break
			with torch.no_grad():
				texts = longclip.tokenize(texts, truncate=True).cuda()
			self.scheduler(step)
			self.mask_net_scheduler(step)
			with (torch.cuda.amp.autocast()):
				loss_dict = self.model(images, texts,  self.rank, soft=args.soft_mask)
				loss = (args.lambda_sparse * (loss_dict['loss_sparsity'] )+
				args.lambda_align* (loss_dict['loss_sidm'] +loss_dict['loss_dism'])
				        )
				loss = loss / self.accumulation_steps
			self.scaler.scale(loss).backward()
			if (i + 1) % self.accumulation_steps == 0 or (i + 1 == len(dataloader)):
				self.scaler.step(self.optimizer)
				self.scaler.step(self.mask_net_optimizer)
				self.scaler.update()
				self.optimizer.zero_grad()
				self.mask_net_optimizer.zero_grad()
			if self.rank == 0:
				pbar.update(1)
				msg = (
							'Epoch: %02d, iter: %05d sidm: %.4f dism: %.4f sparsity: %.4f, num_mean: %d num_min: %d num_max: %d' % (
					epoch, i,
				  loss_dict['loss_sidm'].item(), loss_dict['loss_dism'].item(), loss_dict['loss_sparsity'].item(),
					loss_dict['num_mean'],
					loss_dict['num_min'],
					loss_dict['num_max']
				))
				pbar.set_description(msg)
				if i % 100 == 0:
					print(msg)
					self.loss_manager.log(msg)

			if (i + 1) % 1000 == 0:
				self.test(epoch)
				if self.rank == 0:
					self.save(epoch, self.args.base_model)
					print('LR:   ', self.optimizer.param_groups[0]['lr'],
					      self.mask_net_optimizer.param_groups[0]['lr'])
					result_dict = eval_coco(self.model, self.preprocess)
					print(result_dict)
					msg = 'Epoch: %d, COCO-Short: %s' % (epoch, str(result_dict))
					self.metric_manager.log(msg)

	@torch.no_grad()
	def test_epoch(self, dataloader):
		temp_corr_dict = dict()
		rank = torch.distributed.get_rank()

		for id, (images, text) in enumerate(tqdm(dataloader, disable=(rank != 0))):

			images = images.cuda()
			image_features = self.model.module.encode_image(images)
			image_features = image_features / image_features.norm(dim=-1, keepdim=True)

			text = longclip.tokenize(text, truncate=True).cuda()
			text_feature = self.model.module.encode_text(text)
			text_feature /= text_feature.norm(dim=-1, keepdim=True)

			i = 0
			correct = 0
			total = 0

			for i in range(text_feature.shape[0]):
				text = text_feature[i]
				sim = text @ image_features.T
				sim = sim.squeeze()
				correct_i = torch.argmax(sim)

				if i == correct_i:
					correct = correct + 1
				total = total + 1

		return correct / total

	def test(self, epoch=0):
		rank = torch.distributed.get_rank()
		if rank == 0:
			self.model.eval()
			testset = share4v_val_dataset()

			testloader = torch.utils.data.DataLoader(testset, batch_size=1000, num_workers=4, pin_memory=True)
			with torch.no_grad():
				acc = self.test_epoch(testloader)
				print("=====================================")
				msg = f"test mean of share4v retrieval: {acc} at epoch {epoch}"
				print(msg)
				self.metric_manager.log(msg)
				print("=====================================")

			return

	def save(self, epoch, model_name):
		if isinstance(epoch, int):
			epoch = '%02d' % epoch
		name = "smartclip_epoch%s.pt" % (epoch)
		torch.save(self.model.module.state_dict(), os.path.join(self.target_dir, name))
		print('>>>>> save model <<<<<<', name)

	def train(self, resume=False, warmup_length=200, args=None):
		trainset = share4v_train_dataset()

		train_sampler = DistributedSampler(dataset=trainset, shuffle=True)
		train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, sampler=train_sampler,
		                                           num_workers=8, pin_memory=True)
		if self.rank == 0 and getattr(args, 'output_dir', None):
			# read-only record of the data scale actually used (Full Data Gate evidence)
			import json as _json
			with open(os.path.join(self.target_dir, 'data_scale.json'), 'w') as handle:
				_json.dump({'dataset_size': len(trainset),
				            'steps_per_epoch': len(train_loader),
				            'batch_size_per_gpu': self.batch_size,
				            'world_size': torch.distributed.get_world_size(),
				            'effective_global_batch': self.effective_batch_size,
				            'epochs': self.num_epoch,
				            'dataset_json': os.environ.get('SHARE4V_JSON'),
				            'dataset_root': os.environ.get('SHARE4V_DATA_ROOT')},
				           handle, indent=2, sort_keys=True)
			print('DATASET_SIZE %d STEPS_PER_EPOCH %d EFFECTIVE_GLOBAL_BATCH %d'
			      % (len(trainset), len(train_loader), self.effective_batch_size), flush=True)
		self.scheduler = cosine_lr(self.optimizer, base_lr=self.lr, warmup_length=warmup_length,
		                           steps=self.num_epoch * len(train_loader))
		self.mask_net_scheduler = cosine_lr(self.mask_net_optimizer, base_lr=self.mask_lr, warmup_length=0,
		                                    steps=self.num_epoch * len(train_loader))
		start_epoch = 0
		resume_iter = 0
		for epoch in range(start_epoch, self.num_epoch):
			train_sampler.set_epoch(epoch)
			self.train_epoch(train_loader, epoch, start_iter=resume_iter, args=args)
			if self.rank == 0:
				self.test(epoch)
				self.save(epoch, args.base_model)
				result_dict = eval_coco(self.model, self.preprocess)
				print(result_dict)
				msg = 'Epoch: %d, COCO-Short: %s' % (epoch, str(result_dict))
				self.metric_manager.log(msg)

			if getattr(self, 'max_steps_reached', False):
				break

import torch.distributed as dist


def setup_distributed(backend="nccl", port=None):
	"""Initialize distributed training environment.
	support both slurm and torch.distributed.launch
	see torch.distributed.init_process_group() for more details
	"""
	num_gpus = torch.cuda.device_count()
	print("num_gpus", num_gpus)

	if "SLURM_JOB_ID" in os.environ and False:
		print("SLURM_JOB_ID", os.environ["SLURM_JOB_ID"])
		rank = int(os.environ["SLURM_PROCID"])
		rank2 = dist.get_rank()
		# world_size = int(os.environ["SLURM_NTASKS"])
		world_size = num_gpus
		node_list = os.environ["SLURM_NODELIST"]
		addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
		# specify master port
		if port is not None:
			os.environ["MASTER_PORT"] = str(port)
		elif "MASTER_PORT" not in os.environ:
			os.environ["MASTER_PORT"] = "29522"
		if "MASTER_ADDR" not in os.environ:
			os.environ["MASTER_ADDR"] = addr
		os.environ["WORLD_SIZE"] = str(world_size)
		os.environ["LOCAL_RANK"] = str(rank % num_gpus)
		os.environ["RANK"] = str(rank)
		print("rank, world_size", rank, rank2, world_size, os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"],
		      os.environ['LOCAL_RANK'])
	else:
		rank = int(os.environ["RANK"])
		world_size = int(os.environ["WORLD_SIZE"])
		local_rank = int(os.environ.get("LOCAL_RANK", rank % num_gpus))
		print("rank, world_size", rank, world_size)
	print('>>>>> set device <<<<<<', rank % num_gpus)
	torch.cuda.set_device(rank % num_gpus)
	print('>>>>> init_process_group <<<<<<')
	dist.init_process_group(
		backend=backend,
		world_size=world_size,
		rank=rank,
	)
	print('>>>>> init_process_group done <<<<<<')
	torch.cuda.set_device(device=f'cuda:{rank % num_gpus}')
	print('>>>>> set device done <<<<<<')
	return rank, rank % num_gpus


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='params')
	parser.add_argument('--lr', default=1e-6, type=float, help='lr.')
	parser.add_argument('--mask_lr', default=1e-3, type=float, help='lr for the mask network')
	parser.add_argument('--weight_decay', default=1e-2, type=float, help='wd.')
	parser.add_argument('--log_scale', default=4.6052, type=float, help='clip temperature log scale. disabled')
	parser.add_argument("--exp_name", default="auto", type=str, help="specify experiment name.")
	parser.add_argument("--lambda_sparse", default=2, type=float, help="hyper-parameter for sparsity loss.")
	parser.add_argument("--lambda_align", default=10., type=float, help="hyper-parameter for contrastive loss.")
	parser.add_argument("--soft_mask", default=0, type=float, help="0 for soft mask, 1 for hard mask")
	parser.add_argument("--warmup_length", default=200, type=int, help="warmup_length.")
	parser.add_argument("--base_model", default="L14", help="CLIP Base Model")
	parser.add_argument(
		"--batch-size", type=int, default=256, help="Batch size per gpu."  # 112
	)
	parser.add_argument(
		"--epochs", type=int, default=3, help="Number of epochs to train for."
	)
	parser.add_argument(
		"--resume",
		default=False,
		action='store_true',
		help="resume training from checkpoint."
	)
	parser.add_argument("--download-root", default=None, help="CLIP Base Model download root")
	parser.add_argument('--max_steps', default=None, type=int, help='debug: early-stop after N training iterations (None = official full training)')
	# SmartCLIP reproduction (Phase 3.0A baseline validation): both are opt-in and default to
	# the original behaviour; neither touches the loss, mask, optimizer, scheduler or data.
	parser.add_argument('--output_dir', default=None, type=str,
	                    help='opt-in fixed run directory (default: original ./runs/<auto id>)')
	parser.add_argument('--seed', default=None, type=int,
	                    help='opt-in RNG seed (default: leave the RNGs untouched)')
	parser.add_argument('--init_state', default=None, type=str,
	                    help='opt-in frozen initial state dict to start from (default: the '
	                         'trainer initialises the model as it always did)')
	args = parser.parse_args()
	if args.base_model == 'L14':
		args.base_model = 'ViT-L/14'
	elif args.base_model == 'B16':
		args.base_model = 'ViT-B/16'
	# must happen before the model is built so the initial weights are deterministic
	if args.seed is not None:
		seed_everything(args.seed)
	rank, local_rank = setup_distributed()
	print("DDP Done")

	trainer = CLIP_Clean_Train(
		rank=rank,
		local_rank=local_rank,
		args=args
	)
	trainer.train(resume=args.resume, warmup_length=args.warmup_length, args=args)
