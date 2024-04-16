import torch
import numpy as np
from pytorch_metric_learning.losses import ArcFaceLoss
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu


class DynamicMarginArcFaceLoss(ArcFaceLoss):

    def __init__(self, num_classes, embedding_size, margin=28.6, scale=64, **kwargs):
        """Margin can now be one or as many values as class"""

        super().__init__(num_classes, embedding_size, margin, scale, **kwargs)

    def init_margin(self):

        self.margin = np.radians(self.margin)

        self.dynamic_margin_mode = self.margin.shape != tuple()
        
        if self.dynamic_margin_mode and self.margin.shape != (self.num_classes,):
            raise ValueError("False Dynamic Margin")

        self.margin_tensor = torch.tensor(self.margin, dtype=torch.float32)

    def modify_cosine_of_target_classes(self, cosine_of_target_classes, labels):
        angles = self.get_angles(cosine_of_target_classes)

        # Compute cos of (theta + margin) and cos of theta
        
        margin_tensor = self.margin_tensor.to(cosine_of_target_classes.dtype).to(cosine_of_target_classes.device)

        if self.dynamic_margin_mode:
            
            margin = margin_tensor[labels]
        else:
            margin = margin_tensor

        cos_theta_plus_margin = torch.cos(angles + margin)
        cos_theta = torch.cos(angles)
        
        # Keep the cost function monotonically decreasing
        unscaled_logits = torch.where(
            angles <= np.deg2rad(180) - margin,
            cos_theta_plus_margin,
            cos_theta - margin * torch.sin(margin),
        )

        return unscaled_logits

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        c_f.labels_required(labels)
        c_f.ref_not_supported(embeddings, labels, ref_emb, ref_labels)
        dtype, device = embeddings.dtype, embeddings.device
        self.cast_types(dtype, device)
        miner_weights = lmu.convert_to_weights(indices_tuple, labels, dtype=dtype)
        mask = self.get_target_mask(embeddings, labels)
        cosine = self.get_cosine(embeddings)
        cosine_of_target_classes = cosine[mask == 1]
        # this is the only thing i changed
        modified_cosine_of_target_classes = self.modify_cosine_of_target_classes(
            cosine_of_target_classes,
            labels,
        )
        diff = (modified_cosine_of_target_classes - cosine_of_target_classes).unsqueeze(
            1
        )
        logits = cosine + (mask * diff)
        logits = self.scale_logits(logits, embeddings)
        unweighted_loss = self.cross_entropy(logits, labels)
        miner_weighted_loss = unweighted_loss * miner_weights
        loss_dict = {
            "loss": {
                "losses": miner_weighted_loss,
                "indices": c_f.torch_arange_from_size(embeddings),
                "reduction_type": "element",
            }
        }
        self.add_weight_regularization_to_loss_dict(loss_dict, self.W.t())
        return loss_dict
