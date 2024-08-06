from typing import List
import torch
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)
from typing import List

class Matryoshka_CE_Loss(nn.Module):
	def __init__(self, device, relative_importance: List[float]=None, **kwargs):
		super(Matryoshka_CE_Loss, self).__init__()
		self.criterion = nn.CrossEntropyLoss(**kwargs,)
		# relative importance shape: [G]
		self.relative_importance = relative_importance
		self.device = device

	def forward(self, output, target):
		# output shape: [G granularities, N batch size, C context size] 
		# target shape: [N batch size]

		# Calculate losses for each output and stack them. This is still O(N)
		losses = torch.stack([self.criterion(output_i.view(-1, output_i.size(-1)), target) for output_i in output])
		#logger.info(f"cross entropy loss for logits {output[0][0,:,0]} and targets {target[0,:]} by dimension {losses}")
		
		# Set relative_importance to 1 if not specified
		rel_importance = torch.ones_like(losses) if self.relative_importance is None else torch.tensor(self.relative_importance).to(self.device)
		
		# Apply relative importance weights
		weighted_losses = rel_importance * losses
		return weighted_losses.mean()