import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import wandb

def gather_timestep_coeff(schedule, timesteps, x_shape):
    """
    In diffusion, we keep 1D schedules (e.g., $\sqrt{\bar\alpha_t}$) indexed by
    timestep. This helper picks the coefficient for each item in the batch and
    reshapes it so it can multiply an image/feature tensor of shape `x_shape`.

    To use this function, consider the example
        alpha_t * x
    To compute this in code, we would write
        gather_timestep_coeff(alpha, t, x.shape) * x

    Args:
        schedule: 1D tensor containing values for each timestep.
        timesteps: 1D tensor containing a batch of time indices.
        x_shape: The reference shape.
    Returns:
        The timestep coefficients, reshaped for broadcasting.
    """
    b, *_ = timesteps.shape
    out = schedule.gather(-1, timesteps)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_schedule(timesteps, s=0.008):
    """
    Passes the input timesteps through the cosine schedule for the diffusion process
    Args:
        timesteps: Number of timesteps int. 
        s: The strength of the schedule.
    Returns:
        1D tensor of length timesteps with computed alphas indexed from t = 0 to t = T-1.
    """
    # Define alpha_bar for t in [0, T], then compute alpha_t for t = 0..T-1.
    steps = timesteps
    x = torch.linspace(0, steps, steps + 1)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(alphas, 0.001, 1)


# normalization functions
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# DDPM implementation
class Diffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        channels=3,
        timesteps=1000,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.model = model
        self.num_timesteps = int(timesteps)

        """
        Initializes the diffusion process.
            1. Setup the schedule for the diffusion process.
            2. Define the coefficients for the diffusion process.
        Args:
            model: The model to use for the diffusion process.
            image_size: The size of the images.
            channels: The number of channels in the images.
            timesteps: The number of timesteps for the diffusion process.
        """
        ## TODO: Implement the initialization of the diffusion process ##
        # Pre-compute the coefficients for the diffusion process using scheduler
        # Hint: helpful to define both alpha_bar and alpha_bar_t_minus_1
        # Hint: at t = 0, alpha_bar = 1
        # Hint: what does PyTorch register_buffer do?
        # 1. 📉 Get alphas from the schedule
        alphas = cosine_schedule(self.num_timesteps)
        betas = 1.0 - alphas

        # 2. 📊 Cumulative products: \bar{\alpha}_t
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        # Shift by 1 for \bar{\alpha}_{t-1}, setting the first value (t=0) to 1.0
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # 3. 🎯 Precomputed coefficients for forward diffusion (q_sample)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

        # 4. 🔄 Precomputed coefficients for reverse diffusion (p_sample)
        # Posterior variance: \tilde{\beta}_t = \beta_t * (1 - \bar{\alpha}_{t-1}) / (1 - \bar{\alpha}_t)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )
        self.register_buffer("alphas", alphas)
        self.register_buffer("betas", betas)
        pass
    
    ##############################################
    # FORWARD DIFFUSION    
    ##############################################

    def q_sample(self, x_0, t, noise):
        """
        Applies alpha interpolation between x_0 and noise to simulate sampling
        x_t from the noise distribution.
        Args:
            x_0: The initial images.
            t: 1D tensor containing a batch of time indices to sample at.
            noise: The noise tensor to sample from.
        Returns:
            The sampled images.
        """
        ##### TODO: Implement the q_sample function #######
        # x_t = ...
        # return x_t 

        sqrt_alpha_bar_t = gather_timestep_coeff(
    self.sqrt_alphas_cumprod, t, x_0.shape
)
        sqrt_beta_bar_t = gather_timestep_coeff(
            self.sqrt_one_minus_alphas_cumprod,t, x_0.shape
        )
        scaled_x0 = sqrt_alpha_bar_t * x_0

        scaled_noise = sqrt_beta_bar_t * noise

        x_t = scaled_x0 + scaled_noise
        return x_t
        
    def p_losses(self, x_0, t, noise):
        """
        Computes the loss for the forward diffusion.
        Args:
            x_0: The initial images.
            t: 1D tensor containing a batch of time indices to compute the loss at.
            noise: The noise tensor to use.
        Returns:
            The computed loss.
        """
        ###### TODO: Implement the p_losses function #######
        # define loss function wrt. the model output and the target
        # Hint: you can use pytorch built-in loss functions: F.l1_loss
        x_t = self.q_sample(x_0, t, noise)
        predicted_noise = self.model(x_t, t)
        loss = F.l1_loss(predicted_noise, noise)


        return loss
    
    def forward(self, x_0, noise):
        """
        Acts as a wrapper for p_losses.
        Args:
            x_0: The initial images.
            noise: The noise tensor to use.
        Returns:
            The computed loss.
        """
        b, c, h, w, device, img_size, = *x_0.shape, x_0.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        ##### TODO: Implement the forward function ########
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        loss = self.p_losses(x_0,t, noise)
        
        return loss

    ############################################################
    # BACKWARD DIFFUSION
    ############################################################

    def noise_like(self, shape, device):
        """
        Generates noise with the same shape as the input.
        Args:
            shape: The shape of the noise.
            device: The device on which to create the noise.
        Returns:
            The generated noise.
        """
        noise = lambda: torch.randn(shape, device=device)
        return noise()

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        """
        Computes the (t_index)th sample from the (t_index + 1)th sample using
        the reverse diffusion process.
        Args:
            x: The sampled image at timestep t_index + 1.
            t: 1D tensor of the index of the time step.
            t_index: Scalar of the index of the time step.
        Returns:
            The sampled image at timestep t_index.
        """
        ## TODO: Implement the p_sample function #######
        # sample x_{t-1} from the gaussian distribution wrt. posterior mean and posterior variance
        # Hint: use gather_timestep_coeff function to get the coefficients at time t
        # Hint: use self.noise_like function to generate noise. DO NOT USE torch.randn
        # Begin code here
        pred_noise = self.model(x, t)
        x_0 = (
            x - gather_timestep_coeff(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * pred_noise
        ) / gather_timestep_coeff(self.sqrt_alphas_cumprod, t, x.shape)
        x_0 = x_0.clamp(-1.0, 1.0)
        model_mean = (
            gather_timestep_coeff(self.posterior_mean_coef1, t, x.shape) * x_0
            + gather_timestep_coeff(self.posterior_mean_coef2, t, x.shape) * x
        )
        if t_index == 0:
            outcome = model_mean
        else:
            sigma_t = gather_timestep_coeff(torch.sqrt(self.posterior_variance), t, x.shape)
            noise = self.noise_like(x.shape, x.device)
            outcome = model_mean + sigma_t*noise


        return outcome

    @torch.no_grad()
    def p_sample_loop(self, img):
        """
        Passes noise through the entire reverse diffusion process to generate
        final image samples.
        Args:
            img: The initial noise that is randomly sampled from the noise distribution.
        Returns:
            The sampled images.
        """
        b = img.shape[0]
        #### TODO: Implement the p_sample_loop function ####
        # 1. loop through the time steps from the last to the first
        # 2. inside the loop, sample x_{t-1} from the reverse diffusion process
        # 3. clamp and unnormalize the generated image to valid pixel range
        # Hint: to get time index, you can use torch.full()

        for i in range(self.num_timesteps - 1,-1,-1):
            t = torch.full((b,), i, device=img.device, dtype=torch.long)
            img=self.p_sample(img, t, i)
        img = torch.clamp(img, min=-1.0, max=1.0)
        img = unnormalize_to_zero_to_one(img)

        return img

    @torch.no_grad()
    def sample(self, batch_size):
        """
        Wrapper function for p_sample_loop.
        Args:
            batch_size: The number of images to sample.
        Returns:
            The sampled images.
        """
        #### TODO: Implement the sample function ####
        # Hint: use self.noise_like function to generate noise. DO NOT USE torch.randn
# 1. 📍 Get the device where the model's parameters live
        device = next(self.model.parameters()).device

        # 2. 📐 Define the full 4D image shape: (batch_size, channels, height, width)
        shape = (batch_size, self.channels, self.image_size, self.image_size)

        # 3. 🎲 Generate initial pure Gaussian noise
        img = self.noise_like(shape, device)

        # 4. 🔄 Pass the noise into the reverse diffusion loop
        return self.p_sample_loop(img)
        # ####################################################
