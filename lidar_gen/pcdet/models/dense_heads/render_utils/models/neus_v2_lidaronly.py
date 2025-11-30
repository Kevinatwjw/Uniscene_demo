import torch
from .base_surface_model import SurfaceModel
from functools import partial

import numpy as np
# from threestudio.models.guidance.stable_diffusion_guidance_imgonly import StableDiffusionGuidance
from ..fields.mlp import MLP

class NeuSModelV2LiDAROnly(SurfaceModel):
    def __init__(
        self,
        pc_range,
        voxel_size,
        voxel_shape,
        field_cfg,
        collider_cfg,
        sampler_cfg,
        loss_cfg,
        norm_scene,
        **kwargs
    ):
        super().__init__(
            pc_range=pc_range,
            voxel_size=voxel_size,
            voxel_shape=voxel_shape,
            field_cfg=field_cfg,
            collider_cfg=collider_cfg,
            sampler_cfg=sampler_cfg,
            loss_cfg=loss_cfg,
            norm_scene=norm_scene,
            **kwargs
        )
        self.anneal_end = 50000
        self.pred_intensity = kwargs.get('pred_intensity', False)
        self.pred_raydrop = kwargs.get('pred_raydrop', False)
        if self.pred_intensity or self.pred_raydrop:
            #self.lidar_decoder = nn.Linear(32, 1)
            self.lidar_decoder = MLP(
                in_dim=32,
                layer_width=32,
                out_dim=2 if (self.pred_intensity and self.pred_raydrop) else 1,
                num_layers=3,
                implementation='torch',
                out_activation=None,
                init_bias=-np.log((1 - 0.1) / 0.1)
            )


    def sample_and_forward_field(self, ray_bundle, feature_volume):
        sampler_out_dict = self.sampler(
            ray_bundle,
            occupancy_fn=self.field.get_occupancy,
            sdf_fn=partial(self.field.get_sdf, feature_volume=feature_volume),
            sdf_field=self.field, feature_volume=feature_volume
        )
        ray_samples = sampler_out_dict.pop("ray_samples")
        field_outputs = self.field(ray_samples, feature_volume, return_alphas=True) # ray_samples:  feature_volume: (32, 5, 128, 128)
        weights, _ = ray_samples.get_weights_and_transmittance_from_alphas(
            field_outputs["alphas"]
        )

        samples_and_field_outputs = {
            "ray_samples": ray_samples,
            "field_outputs": field_outputs,
            "weights": weights,
            "sampled_points": ray_samples.frustums.get_start_positions(),
            **sampler_out_dict,
        }
        return samples_and_field_outputs
    
    def get_outputs(self, lidar_ray_bundle, feature_volume, **kwargs):

        lidar_samples_and_field_outputs = self.sample_and_forward_field(
            lidar_ray_bundle, feature_volume
        )
        if self.pred_intensity or self.pred_raydrop:
            weighted_ray_features = (lidar_samples_and_field_outputs['field_outputs']['point_features'] * lidar_samples_and_field_outputs["weights"]).sum(1)
            lidar_relative = self.lidar_decoder(weighted_ray_features)
        else:
            lidar_relative = None

        # lidar
        lidar_ray_samples = lidar_samples_and_field_outputs["ray_samples"]
        lidar_weights = lidar_samples_and_field_outputs["weights"]
        lidar_field_outputs = lidar_samples_and_field_outputs["field_outputs"]
        depth = self.depth_renderer(ray_samples=lidar_ray_samples, weights=lidar_weights)

        lidar_gradients = lidar_field_outputs["gradients"] # torch.Size([21058, 192, 3])
        gradients = lidar_gradients

        outputs = {
            "rgb": None,
            "depth": depth,
            # "weights": weights,
            "sdf": lidar_field_outputs["sdf"],  
            "gradients": gradients,
            "z_vals": lidar_ray_samples.frustums.starts,
            'lidar_relative': lidar_relative
        }

        """ add for visualization"""
        # outputs.update({"sampled_points": samples_and_field_outputs["sampled_points"]})
        # if samples_and_field_outputs.get("init_sampled_points", None) is not None:
        #     outputs.update(
        #         {
        #             "init_sampled_points": samples_and_field_outputs[
        #                 "init_sampled_points"
        #             ],
        #             "init_weights": samples_and_field_outputs["init_weights"],
        #             "new_sampled_points": samples_and_field_outputs[
        #                 "new_sampled_points"
        #             ],
        #         }
        #     )

        # if self.training:
        #     if self.loss_cfg.get("sparse_points_sdf_supervised", False):
        #         sparse_points_sdf, _, _ = self.field.get_sdf(
        #             kwargs["points"].unsqueeze(0), feature_volume
        #         )
        #         outputs["sparse_points_sdf"] = sparse_points_sdf.squeeze(0)

        return outputs

    def forward(self, lidar_ray_bundle, feature_volume, **kwargs):
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """
        lidar_ray_bundle = self.collider(lidar_ray_bundle)  # set near and far
        return self.get_outputs(lidar_ray_bundle, feature_volume, **kwargs)

    def g_loss(self, preds_dict, lidar_targets):
        depth_pred = preds_dict["depth"]
        depth_gt = lidar_targets["depth"]

        loss_dict = {}
        loss_weights = self.loss_cfg.weights

        valid_gt_mask = (depth_gt > 0.0)
        did_return = lidar_targets['did_return'].view(valid_gt_mask.shape)
        valid_gt_mask = valid_gt_mask & did_return
        if loss_weights.get("depth_loss", 0.0) > 0:
            depth_loss = torch.sum(
                valid_gt_mask * torch.abs(depth_gt - depth_pred)
            ) / torch.clamp(valid_gt_mask.sum(), min=1.0)
            loss_dict["depth_loss"] = depth_loss * loss_weights.depth_loss

        # free space loss and sdf loss
        # pred_sdf = preds_dict["sdf"][..., 0]
        # z_vals = preds_dict["z_vals"][..., 0]
        # truncation = self.loss_cfg.sensor_depth_truncation * self.scale_factor

        # front_mask = valid_gt_mask & (z_vals < (depth_gt - truncation))
        # back_mask = valid_gt_mask & (z_vals > (depth_gt + truncation))
        # sdf_mask = valid_gt_mask & (~front_mask) & (~back_mask)

        # if loss_weights.get("free_space_loss", 0.0) > 0:
        #     free_space_loss = (
        #         F.relu(truncation - pred_sdf) * front_mask
        #     ).sum() / torch.clamp(front_mask.sum(), min=1.0)
        #     loss_dict["free_space_loss"] = (
        #         free_space_loss * loss_weights.free_space_loss
        #     )

        # if loss_weights.get("sdf_loss", 0.0) > 0:
        #     sdf_loss = (
        #         torch.abs(z_vals + pred_sdf - depth_gt) * sdf_mask
        #     ).sum() / torch.clamp(sdf_mask.sum(), min=1.0)
        #     loss_dict["sdf_loss"] = sdf_loss * loss_weights.sdf_loss

        # if loss_weights.get("eikonal_loss", 0.0) > 0:
        #     gradients = preds_dict["gradients"]
        #     eikonal_loss = ((gradients.norm(2, dim=-1) - 1) ** 2).mean()
        #     loss_dict["eikonal_loss"] = eikonal_loss * loss_weights.eikonal_loss

        # if self.loss_cfg.get("sparse_points_sdf_supervised", False):
        #     sparse_points_sdf_loss = torch.mean(
        #         torch.abs(preds_dict["sparse_points_sdf"])
        #     )
        #     loss_dict["sparse_points_sdf_loss"] = (
        #         sparse_points_sdf_loss * loss_weights.sparse_points_sdf_loss
        #     )
        # if loss_weights.get("vgg_loss", 0.0) > 0:
        #     if not hasattr(self, "vgg_loss_module"):
        #         self.vgg_loss_module = VGGPerceptualLossPix2Pix()
        #         self.vgg_loss_module.to(rgb_gt.device)
        #     vgg_loss = self.vgg_loss_module(rgb_pred, rgb_gt)
        #     loss_dict["vgg_loss"] = vgg_loss * loss_weights.vgg_loss
        # if loss_weights.get("sds_loss", 0.0) > 0:
        #     if not hasattr(self, "sds_loss_module"):
        #         from threestudio.utils.config import load_config
        #         cfg = load_config("configs/sds/sds.yaml")
        #         self.sds_loss_module = StableDiffusionGuidance(
        #             cfg,
        #         )
        #         self.sds_loss_module
        #         self.sds_loss_module.to(rgb_gt.device)
        #     sds_loss = self.sds_loss_module(rgb_pred, rgb_gt)
        #     loss_dict["sds_loss"] = sds_loss * loss_weights.sds_loss

        if self.pred_intensity and self.pred_raydrop:
            intensity_pred, raydrop_pred = preds_dict['lidar_relative'].split(1, dim=-1)
        elif self.pred_intensity:
            intensity_pred = preds_dict['lidar_relative']
        elif self.pred_raydrop:
            raydrop_pred = preds_dict['lidar_relative']

        if self.pred_intensity:
            intensity_pred = intensity_pred.sigmoid()
            intensity_gt = lidar_targets['intensity']
            intensity_loss = torch.sum(
                #valid_gt_mask * torch.abs(intensity_gt - intensity_pred)
                valid_gt_mask * (intensity_gt - intensity_pred)**2
            ) / torch.clamp(valid_gt_mask.sum(), min=1.0)
            loss_dict["intensity_loss"] = intensity_loss * loss_weights.intensity_loss

        if self.pred_raydrop:
            valid_gt_mask = (depth_gt > 0.0)
            raydrop_loss = torch.nn.functional.binary_cross_entropy_with_logits(raydrop_pred[valid_gt_mask], (~did_return)[valid_gt_mask].to(raydrop_pred), reduction='sum') / torch.clamp(valid_gt_mask.sum(), min=1.0)
            loss_dict['raydrop_loss'] = raydrop_loss * loss_weights.raydrop_loss

        return loss_dict

    def loss(self, preds_dict, lidar_targets):
        return self.g_loss(preds_dict, lidar_targets)