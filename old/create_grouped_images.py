import matplotlib.pyplot as plt
import os

# Your settings
OUTPUTFOLDER = 'multi_lora_training_dress'
OUTPUTFOLDER = 'multi_lora_training_dress_run3/run3'
RESOLUTIONS = [512, 768]
input_images_list = ["input", "input_albu"]  # Adjusted to naming convention
lrs = [1e-5, 1e-4, 1e-3, 5e-5, 5e-6]
steps = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
control_net_options = [False, True]  # ControlNet options

# Ensure output directory for grouped images exists
output_dir = "multi_lora_training_dress_grouped"
os.makedirs(output_dir, exist_ok=True)

# Create subplots for each combination of resolution, input image, and learning rate
for RESOLUTION in RESOLUTIONS:
    for input_images in input_images_list:
        for lr in lrs:
            # 2 rows for True and False options, len(steps) columns for each step count
            fig, axs = plt.subplots(2, len(steps), figsize=(40, 12), sharex=True, sharey=True)  # Adjust figsize as necessary
            fig.subplots_adjust(left=0.01, right=0.95, top=0.9, bottom=0.1, wspace=0.01, hspace=0.1)
            
            for row, USE_CONTROLNET in enumerate(control_net_options):
                for col, step in enumerate(steps):
                    filename = f"output_{RESOLUTION}_{lr}_{input_images}-{step}CONTROL{USE_CONTROLNET}.png"
                    filename = f"output_{RESOLUTION}_{lr}_{input_images}-{step}CONTROL{USE_CONTROLNET}_unet1e-6.png"
                    filepath = os.path.join(OUTPUTFOLDER, filename)
                    ax = axs[row, col]  # Select the appropriate subplot

                    if os.path.exists(filepath):
                        img = plt.imread(filepath)
                        ax.imshow(img)
                        if row == 0:  # Set title only for the first row to avoid redundancy
                            ax.set_title(f"Step {step}")
                    else:
                        ax.set_title(f"Missing {step}")
                    
                    ax.axis('off')  # Hide axes

            # Set suptitle for the group
            plt.suptitle(f"Resolution: {RESOLUTION}, Input: {input_images}, LR: {lr}")

            # Save the figure
            save_path = os.path.join(output_dir, f"output_{RESOLUTION}_{input_images}_{lr}_unet1e-6.png")
            plt.savefig(save_path)
            plt.close(fig)  # Close the figure to free memory
