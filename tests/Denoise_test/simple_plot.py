import numpy as np
import matplotlib.pyplot as plt

def plot_settings(method, avg_score_array, noise_strength, avg_baseline_score):
    ssim_column = avg_score_array[:, 0]
    psnr_column = avg_score_array[:, 1]

    max_ssim = np.max(ssim_column)
    max_psnr = np.max(psnr_column)


    # Find all indices where the value is within tolerance of the max
    ssim_high_indices = np.where(np.abs(ssim_column - max_ssim) == 0)[0]
    psnr_high_indices = np.where(np.abs(psnr_column - max_psnr) == 0)[0]

    max_ssim_index = np.argmax(ssim_column)
    max_psnr_index = np.argmax(psnr_column)

    ssim_indices = np.arange(len(ssim_column))
    psnr_indices = np.arange(len(psnr_column))

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    fig.suptitle(f"{method} @{(noise_strength**0.5):.3f} std, {noise_strength} var")

    # SSIM plot
    #axs[0].plot(ssim_column, ".")
    
    if method == "gaussian_blur":
        kernel_column = avg_score_array[:, 2]
        sigma_column = avg_score_array[:, 3]

        min_sigma = np.min(sigma_column)
        max_sigma = np.max(sigma_column)
        # Find all indices where the value is within tolerance of the max
        max_sigma_indices = np.where(np.abs(sigma_column - max_sigma) == 0)[0]
        min_sigma_indices = np.where(np.abs(sigma_column - min_sigma) == 0)[0]

      
        
        for i in range(len(max_sigma_indices)):
            max_sigma_index = max_sigma_indices[i]
            min_sigma_index = min_sigma_indices[i]

        
            axs[0].plot(
                ssim_indices[min_sigma_index:max_sigma_index],
                ssim_column[min_sigma_index:max_sigma_index]
                )
            
            axs[1].plot(
                psnr_indices[min_sigma_index:max_sigma_index],
                psnr_column[min_sigma_index:max_sigma_index]
                )

        #SSIM plot
        for idx in ssim_high_indices:
            axs[0].axvline(x=idx, color='r', linestyle='--', label=f'| Kernel_size: {kernel_column[idx]} | Sigma: {round(sigma_column[idx], 3)} |')

        #PSNR plot
        for idx in psnr_high_indices:
            axs[1].axvline(x=idx, color='blue', linestyle='--', label=f'| Kernel_size: {kernel_column[idx]} | Sigma: {round(sigma_column[idx], 3)} |')

    elif method == "billateral":
        diameter_column = avg_score_array[:, 2]
        sigma_colour_column = avg_score_array[:, 3]
        sigma_space_column = avg_score_array[:, 4]

        min_sigma_space = np.min(sigma_space_column)
        max_sigma_space = np.max(sigma_space_column)
        # Find all indices where the value is within tolerance of the max
        max_sigma_space_indices = np.where(np.abs(sigma_space_column - max_sigma_space) == 0)[0]
        min_sigma_space_indices = np.where(np.abs(sigma_space_column - min_sigma_space) == 0)[0]

      
        for i in range(len(max_sigma_space_indices)):
            max_sigma_index = max_sigma_space_indices[i]
            min_sigma_index = min_sigma_space_indices[i]

        
            axs[0].plot(
                ssim_indices[min_sigma_index:max_sigma_index],
                ssim_column[min_sigma_index:max_sigma_index]
                )
            
            axs[1].plot(
                psnr_indices[min_sigma_index:max_sigma_index],
                psnr_column[min_sigma_index:max_sigma_index]
                )

        #SSIM plot
        for idx in ssim_high_indices:
            axs[0].axvline(x=idx, color='r', linestyle='--', label=f'| Diameter: {round(diameter_column[idx])} | S_Color: {round(sigma_colour_column[idx])} | S_Space: {round(sigma_space_column[idx])} |')
        
        #PSNR plot
        for idx in psnr_high_indices:
            axs[1].axvline(x=idx, color='blue', linestyle='--', label=f'| Diameter: {round(diameter_column[idx])} | S_Color: {round(sigma_colour_column[idx])} | S_Space: {round(sigma_space_column[idx])} |')

    elif method == "median_blur":
        ksize_column = avg_score_array[:, 2]

        #SSIM plot
        for idx in ssim_high_indices:
            axs[0].axvline(x=idx, color='r', linestyle='--', label=f'| Kernel_size:  {round(ksize_column[idx])} |')
    
        #PSNR plot
        for idx in psnr_high_indices:
            axs[1].axvline(x=idx, color='blue', linestyle='--', label=f'| Kernel_size:  {round(ksize_column[idx])} |')
       
    elif method == "fastnlmeans":
        h_column = avg_score_array[:, 2]
        h_color_column = avg_score_array[:, 3]
        template_size_column = avg_score_array[:, 4]
        search_size_column = avg_score_array[:, 5]

        min_search_size = np.min(search_size_column)
        max_search_size = np.max(search_size_column)
        # Find all indices where the value is within tolerance of the max
        max_search_size_indices = np.where(np.abs(search_size_column - max_search_size) == 0)[0]
        min_search_size_indices = np.where(np.abs(search_size_column - min_search_size) == 0)[0]

      
        for i in range(len(max_search_size_indices)):
            max_index = max_search_size_indices[i]
            min_index = min_search_size_indices[i]

        
            axs[0].plot(
                ssim_indices[min_index:max_index],
                ssim_column[min_index:max_index]
                )
            
            axs[1].plot(
                psnr_indices[min_index:max_index],
                psnr_column[min_index:max_index]
                )


        #SSIM plot
        for idx in ssim_high_indices:
            axs[0].axvline(x=idx, color='r', linestyle='--', label=f'| H: {round(h_column[idx])} | H_colour: {round(h_color_column[idx])} | Template: {round(template_size_column[idx])} | Search: {round(search_size_column[idx])} |')
        
        #PSNR plot
        for idx in psnr_high_indices:
            axs[1].axvline(x=idx, color='blue', linestyle='--', label=f'| H: {round(h_column[idx])} | H_colour: {round(h_color_column[idx])} | Template: {round(template_size_column[idx])} | Search: {round(search_size_column[idx])} |')
    

    axs[0].axhline(y=ssim_column[max_ssim_index], color='red', linestyle=':', label=f'Max SSIM value={ssim_column[max_ssim_index]:.4f}')
    axs[0].axhline(y=avg_baseline_score[0], color='black', linestyle=':', label=f'Baseline Value={avg_baseline_score[0]:.4f}')
    axs[0].set_ylabel("SSIM")
    axs[0].set_ylim(0.80, 1)

    # PSNR plot
    axs[1].axhline(y=psnr_column[max_psnr_index], color='blue', linestyle=':', label=f'Max PSNR value={psnr_column[max_psnr_index]:.4f}')
    axs[1].axhline(y=avg_baseline_score[1], color='black', linestyle=':', label=f'Baseline value={avg_baseline_score[1]:.4f}')
    axs[1].set_ylabel("PSNR")
    axs[1].set_xlabel("Total Steps")
    axs[1].set_ylim(25, 46)

    # Collect handles and labels from both axes
    handles0, labels0 = axs[0].get_legend_handles_labels()
    handles1, labels1 = axs[1].get_legend_handles_labels()

    # Pad the shorter list with empty entries so both have the same length
    max_len = max(len(handles0), len(handles1))
    while len(handles0) < max_len:
        handles0.append(plt.Line2D([], [], color='none'))
        labels0.append("")
    while len(handles1) < max_len:
        handles1.append(plt.Line2D([], [], color='none'))
        labels1.append("")

    # Concatenate for two columns: first all SSIM, then all PSNR
    handles = handles0 + handles1
    labels = labels0 + labels1

    # Place a single legend at the bottom center, two columns
    fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02-0.03*((len(labels)+1)//2)),
        ncol=2,
        fontsize='small', 
        columnspacing=2.0
    )

    #save the figure
    #plt.savefig(f"tests/plots/{method}_{noise_strength}.pdf", format='pdf', dpi=2400, bbox_inches='tight')
    plt.show()

def main():
    denoise_list = ["gaussian", "bilateral", "median", "fastnlmeans"]
    denoise_list = ["fastnlmeans"]

    for denoise_type in denoise_list:
        noise_strength = [5, 10, 15, 20]

        if denoise_type == "gaussian":
            path = "tests/gaussian_test_files/"
            method_list = ['gaussian_blur',]

            """if denoise_type == "guassian":
            path = "guassian_test_files/"
            #method_list = ['denoise', 'unsharp', 'high_pass']
            method_list = ['denoise'] """
        
        elif denoise_type == "bilateral":
            path = "tests/bilateral_test_files/"
            method_list = ['billateral']

        elif denoise_type == "median":
            path = "tests/median_test_files/"
            method_list = ['median_blur']

        elif denoise_type == "fastnlmeans":
            path = "tests/fastnlmeans_test_files/"
            method_list = ['fastnlmeans']

        for method in method_list:
            baseline_score_array = np.load("tests/baseline_score.npy")

            avg_baseline_score = np.zeros(2)
            avg_baseline_score[0] = np.average(baseline_score_array[:, 0])
            avg_baseline_score[1] = np.average(baseline_score_array[:, 1])

            for noise in noise_strength:
                with open(f"{path}{method}_estimate_noise_{noise}.npy", "rb") as f:
                    noise_estimate_array = np.load(f)
                    avg_noise = np.average(noise_estimate_array[:, 1])
                    real_noise = np.average(noise_estimate_array[:, 0])
            
                    #print(f"Real noise: {real_noise}, Estimated noise: {avg_noise}, procentage error: {abs(real_noise - avg_noise) / real_noise * 100:.2f}%")


                with open(f"{path}{method}_avg_score_{noise}.npy", "rb") as f:
                    avg_score_array = np.load(f)

                    plot_settings(method, avg_score_array, noise, avg_baseline_score)


if __name__ == "__main__":
    main()

