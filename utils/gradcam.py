import torch
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        handle_fw = self.target_layer.register_forward_hook(forward_hook)
        handle_bw = self.target_layer.register_backward_hook(backward_hook)
        self.hook_handles.extend([handle_fw, handle_bw])

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        loss = output[0, class_idx]
        loss.backward(retain_graph=True)

        gradients = self.gradients[0]  # C x H x W
        activations = self.activations[0]  # C x H x W

        weights = torch.mean(gradients, dim=(1, 2))  # global avg pool

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = cam.cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.size(3), input_tensor.size(2)))
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam