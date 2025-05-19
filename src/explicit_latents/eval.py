from torch.utils import data
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.slot_attention.autoencoder import SlotAttentionAutoEncoder
from src.explicit_latents.autoencoder import ExplicitLatentAutoEncoder
from src.utils import IMG_CHANNELS, ImageDataset, get_config_argument, load_config, set_seed, plot_images, DEVICE

PERTURBATION_MAGNITUDE = 0.5
NUM_OUTPUT_FIGS = 5
OUTPUT_DIR = "data/figures/"


def main():
    config_name = get_config_argument()
    config = load_config(config_name)
    config_SA = config["slot_attention"]
    config_latent = config["explicit_latents"]
    
    set_seed(config_latent["seed"])
    ckpt_path_SA = config_SA["ckpt_path"]
    ckpt_path_disentangle = config_latent["ckpt_path"]

    print("Loading model:", ckpt_path_SA)
    model_SA = SlotAttentionAutoEncoder(
        resolution=config_SA["resolution"],
        num_slots=config_SA["num_slots"],
        num_iterations=config_SA["num_iterations"], 
        num_channels=IMG_CHANNELS,
        slots_dim=config_SA["slots_dim"], 
        encdec_dim=config_SA["encdec_dim"]).to(DEVICE)
    model_SA.eval() 
    missing_keys, unexpected_keys = model_SA.load_state_dict(torch.load(ckpt_path_SA, weights_only=True)['model_state_dict'], strict=False)
    
    # these keys can be generated again by the model
    generatable_keys = ['encoder_cnn.encoder_pos.grid', 'decoder_cnn.decoder_pos.grid']
    for key in generatable_keys:
        if key in missing_keys:
            missing_keys.remove(key)
    
    if missing_keys:
        raise KeyError(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        raise KeyError(f"Unexpected keys: {unexpected_keys}")

    print("Loading model:", ckpt_path_disentangle)
    model_disentangle = ExplicitLatentAutoEncoder(
        latent_dim=config_latent["latent_dim"], 
        slots_dim=config_SA["slots_dim"]
    ).to(DEVICE)
    model_disentangle.eval()
    model_disentangle.load_state_dict(torch.load(ckpt_path_disentangle, weights_only=True)['model_state_dict'], strict=True)

    print(f"Loading observation test dataset: {config_SA['test_path']}")
    test_dataset = ImageDataset(hdf5_file=config_SA["test_path"], hdf5_format=config_SA["hdf5_format"])
    test_dataloader = data.DataLoader(test_dataset, batch_size=config_SA['batch_size'], shuffle=False, drop_last=False)
    print(f"Number of test samples: {len(test_dataset) * config_SA['batch_size']}")

    with torch.no_grad():
        explicit_loss_list = []
        SA_loss_list = []
        SA_explicit_loss_list = []

        for batch_idx, batch in enumerate(test_dataloader):
            obs_true = batch.to(DEVICE)
            # ENCODE
            obs_recon_SA, _, _, slots_active_true, slot_background = model_SA(obs_true)
            slots_active_recon, z = model_disentangle(slots_active_true)

            # DECODE
            slots_recon_all = torch.cat((slots_active_recon, slot_background), dim=1)
            obs_recon_explicit, _, _ = model_SA.decode(slots_recon_all)

            criterion = torch.nn.MSELoss()

            for i in range(len(obs_true)):
                explicit_loss_list.append(criterion(slots_active_recon[i], slots_active_true[i]).item())
                SA_loss_list.append(criterion(obs_recon_SA[i], obs_true[i]).item())
                SA_explicit_loss_list.append(criterion(obs_recon_explicit[i], obs_true[i]).item())

            if batch_idx == 0:
                object_index = 0
                recon_perturbed_list = []
                
                for l in range(config_latent['latent_dim']):
                    z_perturbed = z.clone()
                    z_perturbed[:, object_index, l] += PERTURBATION_MAGNITUDE
                    slots_active_recon_perturbed = model_disentangle.decode(z_perturbed)
                    all_slots_perturbed = torch.concat((slots_active_recon_perturbed, slot_background), dim=1)
                    recon_perturbed, _, _ = model_SA.decode(all_slots_perturbed)
                    recon_perturbed_list.append(recon_perturbed)

                print("--- z's ---")
                
                for i in range(NUM_OUTPUT_FIGS):
                    print(f"z_{i}:\n{z[i].cpu().numpy()}")
                print("-----------")
                
                for i in range(NUM_OUTPUT_FIGS):
                    # plot image row: original, reconstructed (SA), reconstructed (from latent dim), reconstructions with latent perturbations
                    imgs = [obs_true[i], obs_recon_SA[i], obs_recon_explicit[i]]
                    loss = criterion(slots_active_true[i], slots_active_recon[i])
                    loss_str = f"{loss.item():.6f}".replace('.', '') # remove decimal point for filename
                    labels = ["Original", "Reconstructed (SA)", "Reconstructed (from latent dim)"]
                    for j, recon_perturbed in enumerate(recon_perturbed_list):
                        imgs.append(recon_perturbed[i])
                        labels.append(f"Pert #{j}")
                    save_path = f"{OUTPUT_DIR}explicit_{loss:.2E}.png"
                    plot_images(imgs, save_path, labels=labels)
        
        # scatter plot with logarithmic axes
        fig, ax = plt.subplots()
        ax.scatter(explicit_loss_list, SA_explicit_loss_list, marker='x')
        
        # Add additional datapoint
        avg_explicit_loss = np.mean(explicit_loss_list)
        avg_SA_explicit_loss = np.mean(SA_explicit_loss_list)
        ax.scatter(avg_explicit_loss, avg_SA_explicit_loss, marker='o', color='red', label='Average')

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Slot reconstruction loss (log scale)")
        ax.set_ylabel("Observation reconstruction loss (log scale)")
        ax.legend()
        plt.savefig(f"{OUTPUT_DIR}explicit_vs_SA.png")
        plt.close(fig)

        print()
        print("--- Slot Attention ---")
        print(f"Average loss: {np.mean(SA_loss_list):.8f}")
        print(f"Standard deviation of loss: {np.std(SA_loss_list):.8f}")
        print("--- Explicit Autoencoder ---")
        print(f"Average loss: {np.mean(explicit_loss_list):.8f}")
        print(f"Standard deviation of loss: {np.std(explicit_loss_list):.8f}")
        print("--- Slot Attention + Explicit Autoencoder ---")
        print(f"Average loss: {np.mean(SA_explicit_loss_list):.8f}")
        print(f"Standard deviation of loss: {np.std(SA_explicit_loss_list):.8f}")


if __name__ == '__main__':
    main()