import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

from pyutils.pyselect import Select
from pyutils.pyvector import Vector

import zfit

class Compare():
    """Class to conduct comparisons between cut or data sets
    """
    def __init__(self ):
      """
      """
      
      # Custom prefix for log messages from this processor
      self.print_prefix = "[Compare] "
      print(f"{self.print_prefix}Initialised")

    def plot_variable(self, val_overlay, val_label, filenames, lo, hi, cut_lo, cut_hi, mc_count, columns=[], nbins = 50, density=True):
      """
      Plots distributions of the given parameter (val), splitting by process code

      Args:
          val : list of values e.g. rmax
          val_label : text formated value name e.g. "rmax"
          lo : plot range lower bound
          hi : plot range upper bound
          cut_lo : lower cut choice
          cut_hi : upper cut choice
          mc_counts : list of process codes

      Returns:
          plots saved as pdfs
      """
      sets = []
      rpc = ["e+", "e-"]
      fig, ax1 = plt.subplots(1,1)
      cols = ['blue','green']
      labs = ['irpc','erpc']
      styles = ['bar','step','step']
      lines=["","-","--"]
      alphas = [0.2,1,1]
      text_contents = []
      for i, val in enumerate(val_overlay):
        val = ak.drop_none(val)
        #val_cosmics = val.mask[mc_count[i] == -1]
        #val_cosmics = np.array(ak.flatten(val_cosmics, axis=None))
        val_erpc = val.mask[mc_count[i] == 178]
        val_erpc = np.array(ak.flatten(val_erpc,axis=None))
        val_irpc = val.mask[mc_count[i] == 179]
        val_irpc = np.array(ak.flatten(val_irpc,axis=None))
        mean_val = np.mean(val_irpc)
        std_dev = np.std(val_irpc)
        text_contents.append(str(rpc[i])+ f"Mean: {mean_val:.2f}\nStd Dev: {std_dev:.2f}")
        sets.append([val_irpc,val_erpc])
      for i in range(0,len(sets)):
        ax1.set_yscale('log')
        dummy_handle = ax1.plot([], marker="",color='white', label=columns[i])
        n, bins, patch = ax1.hist(sets[i],range=(lo,hi), color=cols, label=labs, bins=nbins, histtype=styles[i], alpha=alphas[i], stacked=True, density=density)


      ax1.set_xlabel(str(val_label))
      ax1.set_xlim(lo,hi)
 
      ax1.legend(ncol=len(columns))#,loc='upper center')
      for i in range(0,len(text_contents)):
        plt.text(0.1, 0.95-i*0.15, text_contents[i], 
                 transform=plt.gca().transAxes,
                 fontsize=10,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))

      plt.savefig(str(filenames)+"_selection.pdf")
      plt.show()
      
    

    def compare_resolution(self, recomom, truemom):
      """
      stores difference between recon and true momentum for resolution comparison
      """
      truemom = truemom.mask[truemom > 85] # removes anything that we dont care about on the reconstruction
      recomom = ak.nan_to_none(recomom)
      recomom = ak.drop_none(recomom)
      truemom = ak.nan_to_none(truemom)
      truemom = ak.drop_none(truemom)

      differences = [
        reco[0] - truemom[i][j][0]
        for i, reco_list in enumerate(recomom)
        for j, reco in enumerate(reco_list)
        if len(reco) != 0 and len(truemom[i][j]) != 0
      ]
      
      return differences

    def plot_resolution(self, val_overlay, val_label, filenames, lo, hi, columns=[], density=True):
      """
      Plots distributions of the given parameter (val), splitting by process code

      Args:
          val : list of values e.g. rmax
          val_label : text formated value name e.g. "rmax"
          lo : plot range lower bound
          hi : plot range upper bound

      Returns:
          plots saved as pdfs
      """
      fig, (ax1) = plt.subplots(1,1)
      sets=[]
      cols = ['blue']
      labs = ['e+','e-']
      styles = ['bar','step']
      lines=["","-"]
      alphas = [0.2,1]
      text_contents = []
      for i, val in enumerate(val_overlay):
        val = ak.drop_none(val)
        val = np.array(ak.flatten(val,axis=None))
        mean_val = np.mean(val)
        std_dev = np.std(val)
        text_contents.append(str(labs[i])+ f"Mean: {mean_val:.2f}\nStd Dev: {std_dev:.2f}")
        sets.append([val])

      for i in range(0,len(sets)):
        ax1.set_yscale('log')
        dummy_handle = ax1.plot([], marker="",color='white', label=columns[i])
        n, bins, patch = ax1.hist(sets[i],range=(lo,hi), color=cols, label=labs, bins=50, histtype=styles[i], alpha=alphas[i], stacked=True, density=density)

      ax1.set_xlabel(str(val_label))
      ax1.set_xlim(lo,hi)
      ax1.legend(ncol=len(columns))
      for i in range(0,len(text_contents)):
        plt.text(0.1, 0.95-i*0.1, text_contents[i], 
                 transform=plt.gca().transAxes,
                 fontsize=10,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))

      
      plt.savefig(str(filenames)+"_resolution.pdf")
      plt.show()
      
    def plot_particle_counts(self, mc_counts, columns):
      """
      Plot a grouped horizontal bar chart comparing particle type counts
      between different datasets and adds percentage change labels.
      
      Args:
          mc_counts : list of arrays/lists of particle codes (one per dataset)
          columns   : labels for datasets (e.g., ["old_cuts", "no_cuts"])
      """
      # Map PDG/startCodes to categories
      labels = ["DIO", "IPA", "CEMLL", "CEPLL", "eRPC", "iRPC", "eRMC", "iRMC", "Cosmic", "Other"]
      pdg_codes = [166, 114, 168, 176, 178, 179, 171, 172, -1, -2]
      num_categories = len(pdg_codes)
      num_datasets = len(mc_counts)

      # Use NumPy's vectorized operations for efficient counting
      datasets = np.zeros((num_datasets, num_categories), dtype=int)
      for i, mc in enumerate(mc_counts):
          if mc is not None and len(mc) > 0:
              mc_array = np.array(mc)
              for j, code in enumerate(pdg_codes):
                  datasets[i, j] = np.sum(mc_array == code)
      
      # Check that there are at least two datasets for a comparison
      if num_datasets < 2:
          print("Not enough datasets for percentage change calculation. Plotting without it.")
          # Re-run the original plotting logic if needed
          # ...
          return

      # Calculate percentage change based on the first dataset
      # Avoids division by zero by setting change to 0 if the original value is 0
      with np.errstate(divide='ignore', invalid='ignore'):
          old_counts = datasets[0]
          new_counts = datasets[1]
          percent_changes = ((new_counts - old_counts) / old_counts) * 100
          percent_changes[np.isinf(percent_changes) | np.isnan(percent_changes)] = 0

      # Plot grouped horizontal bars
      y = np.arange(num_categories)
      bar_height = 0.8 / num_datasets
      
      fig, ax = plt.subplots(figsize=(12, 6))

      bars = []
      for i, data in enumerate(datasets):
          bars.append(ax.barh(y + i * bar_height, data, height=bar_height, label=columns[i]))
      
      # Add percentage change labels to the second set of bars
      for i, bar in enumerate(bars[1]): # Iterate over the bars of the second dataset
          # Get the percentage change for the corresponding category
          change = percent_changes[i]
          
          # Format the label string
          label_text = f'{change:.1f}%'
          
          # Choose color based on whether change is positive or negative
          color = 'red' if change < 0 else 'green'
          
          # Position the label
          # Get the y-position and width (x-value) of the bar
          ax.text(
              bar.get_width(), 
              bar.get_y() + bar.get_height() / 2, 
              label_text, 
              ha='left', 
              va='center',
              color=color,
              fontsize=8
          )

      # Center the y-tick labels correctly
      ax.set_yticks(y + bar_height * (num_datasets - 1) / 2)
      ax.set_yticklabels(labels)
      ax.set_xlabel("Event counts")
      ax.set_title("Comparison of particle types with Percentage Change")
      #ax.set_xlim(0, 60000)
      ax.legend()
      
      plt.tight_layout()
      plt.savefig("particle_comparison_with_changes.pdf")
      plt.show()
      
    def plot_cut_eff(self, numerator_array, denominator_array, bin_centers, title="Ratio of Arrays", name="all", x_label="true momentums", y_label="Efficiency"):
      """
      Calculates the element-wise ratio of two arrays and plots the result.
      
      Args:
          numerator_array (np.ndarray): The array for the numerator.
          denominator_array (np.ndarray): The array for the denominator.
          title (str): The title of the plot.
          x_label (str): The label for the x-axis.
          y_label (str): The label for the y-axis.
      """
      # 1. Ensure arrays have the same shape
      if numerator_array.shape != denominator_array.shape:
          raise ValueError("Input arrays must have the same shape.")

      # 2. Handle potential division by zero
      # Use np.divide with 'where' to perform division only where denominator is not zero.
      # Specify the `out` array to hold the result and set values to 0 where denominator is zero.
      ratio = np.divide(numerator_array, denominator_array, out=np.zeros_like(numerator_array, dtype=float), where=denominator_array != 0)

      # 3. Create the plot
      fig, ax = plt.subplots(figsize=(10, 6))
      
      #ax.plot(ratio, marker='o', linestyle='-', color='b')
      #plt.scatter(, marker="-")
      plt.plot(bin_centers, ratio, marker='o', linestyle='-', label='Connected Points')
      ax.set_title(title)
      ax.set_xlabel(x_label)
      ax.set_ylabel(y_label)
      ax.grid(True)
      
      # Optional: Highlight where denominator was zero
      zero_indices = np.where(denominator_array == 0)[0]
      #if zero_indices.size > 0:
      #    ax.plot(zero_indices, ratio[zero_indices], 'rx', label='Denominator was zero')
      #    ax.legend()
      
      
      plt.savefig("eff_"+str(name)+".pdf")
      plt.show()
      
    def plot_2D(self, xs, ys):
      
      for i, x in enumerate(xs):
        y = ys[i]
        x_sync = []
        y_sync = []
        for j, element in enumerate(x):
          for k, subelement in enumerate (element):
            for l, subsubelement in enumerate (subelement):
              if(x[j][k][l] != None):
                for m, y_els in enumerate (y[j][k]):
                  if(y[j][k][m] != None):
                    
                    x_sync.append(x[j][k][l])
                    y_sync.append(y[j][k][m])
              

        # Plot the 2D histogram
        fig, ax = plt.subplots(figsize=(8, 6))
        h = ax.hist2d(y_sync, x_sync, bins=50, cmin = 1, cmap='viridis')

        # Add labels and a color bar
        ax.set_xlabel("True Momentum at TrkEnt [MeV/c]")
        ax.set_ylabel("rmax")
       
        fig.colorbar(h[3], ax=ax, label='Counts in bin') # Changed from h to h[3] to match hist2d output

        # Display the plot
        plt.show()

    def fit_momentum(self, data_list):
        """
        Fits a simple Gaussian shape to the reconstructed momentum data
        using an extended unbinned maximum likelihood fit.
        """
        # Create figure with two subplots: main plot and ratio plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        colors = ["green", "blue"]
        labels = ["e+", "e-"]
        
        # Store text box y-positions to avoid overlap
        text_y_pos = [0.3, 0.5]
        mean = 0.
        mean_err = 0.
        sigma = 0.
        sigma_err = 0.
        norm = 0.
        for i, data in enumerate(data_list):
            mom_mag_skim = ak.nan_to_none(data)
            mom_mag_skim = ak.drop_none(mom_mag_skim)

            # Define the observable space for the fit
            obs_mom = zfit.Space('x', limits=(95, 115))
            mom_np = ak.to_numpy(ak.flatten(mom_mag_skim, axis=None))
            mom_zfit = zfit.Data.from_numpy(array=mom_np, obs=obs_mom)
            
            # Define parameters for the Gaussian shape and yield
            mu = zfit.Parameter("mu", 98, 96, 102)
            sigma = zfit.Parameter("sigma", 10, 5, 20)
            N_RPC = zfit.Parameter('N_RPC', 250000, 100, 400000)

            # Create the extended Gaussian PDF
            gauss = zfit.pdf.Gauss(obs=obs_mom, mu=mu, sigma=sigma, extended=N_RPC)
            
            # Create the extended unbinned negative log-likelihood loss
            nll = zfit.loss.ExtendedUnbinnedNLL(model=gauss, data=mom_zfit)
            
            # Minimize the loss and get the result
            minimizer = zfit.minimize.Minuit()
            result = minimizer.minimize(loss=nll)
            hesse_errors = result.hesse()
            print(result)
            
            # --- Plotting the fit result ---
            
            fit_range = (obs_mom.lower[0, 0], obs_mom.upper[0, 0])
            n_bins = 50
            bin_width = (fit_range[1] - fit_range[0]) / n_bins
            
            # --- Main plot ---
            
            mom_plot = np.linspace(fit_range[0], fit_range[1], 200).reshape(-1, 1)

            gauss_curve = zfit.run(gauss.pdf(mom_plot) * result.params[N_RPC]['value'] * bin_width)
            ax1.plot(mom_plot.flatten(), gauss_curve.flatten(), color=colors[i], linestyle="--", label=str(labels[i])+' Fitted Gaussian')
            ax1.grid(True)
            
            data_hist, data_bins, _ = ax1.hist(mom_np, color=colors[i], bins=n_bins, range=fit_range, label=labels[i], histtype='step')
            data_bin_center = (data_bins[:-1] + data_bins[1:]) / 2
            ax1.errorbar(data_bin_center, data_hist, yerr=np.sqrt(data_hist), fmt='.', color=colors[i], capsize=2)
            
            ax1.set_xlabel('Reconstructed Momentum [MeV/c]')
            ax1.set_ylabel('# of events per bin')
            ax1.legend()
            ax1.set_title('Gaussian Fit to Momentum Data (Extended Unbinned)')
            
            # --- Add text box with fit parameters ---
            param_text = (
                f"Fit parameters for {labels[i]}:\n"
                f"$\\mu = {result.params[mu]['value']:.2f} \\pm {hesse_errors[mu]['error']:.2f}$ \n"
                f"$\\sigma = {result.params[sigma]['value']:.2f} \\pm {hesse_errors[sigma]['error']:.2f}$\n"
                f"$N_{{RPC}} = {result.params[N_RPC]['value']:.0f} \\pm {hesse_errors[N_RPC]['error']:.2f}$"
            )
            
            props = dict(boxstyle='round', facecolor=colors[i], alpha=0.3)
            
            # Position the text box in the upper left corner of the subplot
            # with an offset for each iteration
            ax1.text(0.4, text_y_pos[i], param_text, transform=ax1.transAxes,
                     fontsize=10, verticalalignment='top', bbox=props)
            
            # --- Ratio plot ---
            
            data_bin_center_2d = data_bin_center.reshape(-1, 1)
            fit_at_bin_center = zfit.run(gauss.pdf(data_bin_center_2d) * result.params[N_RPC]['value'] * bin_width)
            ratio = data_hist / fit_at_bin_center
            
            ax2.errorbar(data_bin_center, ratio, yerr=np.sqrt(data_hist) / fit_at_bin_center, fmt='.', color=colors[i], capsize=2)
            ax2.axhline(1, color='gray', linestyle='--')
            ax2.set_ylabel('Ratio (Data/Fit)')
            ax2.set_xlabel('Reconstructed Momentum [MeV/c]')
            ax2.set_ylim(0.8, 1.2)
            ax2.grid(True)
            mean = result.params[mu]['value']
            #meanr_err = hesse_errors[mu]['error']
            sigma = result.params[sigma]['value']
            #sigma_err = hesse_errors[sigma]['error']
            norm = result.params[N_RPC]['value']
        plt.tight_layout()
        plt.savefig("RPCfit.pdf")
        plt.show()
        return mean, mean_err, sigma, sigma_err, norm

    def overlay_fit(self, mean, mean_err, sigma, sigma_err, norm, data_list, mc_count):
        """
        Fits a simple Gaussian shape to the reconstructed momentum data
        using an extended unbinned maximum likelihood fit.
        """
        # Create figure with two subplots: main plot and ratio plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        colors = ["black"]
        labels = ["MDS2c"]
        
        # Store text box y-positions to avoid overlap
        text_y_pos = [0.8] 

        for i, data in enumerate(data_list):
            mom_mag_skim = ak.nan_to_none(data)
            mom_mag_skim = ak.drop_none(mom_mag_skim)
            
            true_rpc = mom_mag_skim.mask[(mc_count[i] == 999) ]
            true_rpc = ak.to_numpy((ak.flatten(true_rpc,axis=None)))
            print(true_rpc)
            print(mc_count)

            # Define the observable space for the fit
            obs_mom = zfit.Space('x', limits=(95, 115))
            mom_np = ak.to_numpy(ak.flatten(mom_mag_skim, axis=None))
            mom_zfit = zfit.Data.from_numpy(array=true_rpc, obs=obs_mom)
            
            # Define parameters for the Gaussian shape and yield
            mu = zfit.Parameter("mu", mean, floating=False)
            sigma = zfit.Parameter("sigma", sigma, floating=False)
            N_RPC = zfit.Parameter('N_RPC', norm, norm-0.05*norm, norm+0.05*norm)

            # Create the extended Gaussian PDF
            gauss = zfit.pdf.Gauss(obs=obs_mom, mu=mu, sigma=sigma, extended=N_RPC)
            
            # Create the extended unbinned negative log-likelihood loss
            nll = zfit.loss.ExtendedUnbinnedNLL(model=gauss, data=mom_zfit)
            
            # Minimize the loss and get the result
            minimizer = zfit.minimize.Minuit()
            result = minimizer.minimize(loss=nll)
            hesse_errors = result.hesse()
            print(result)
            
            # --- Plotting the fit result ---
            
            fit_range = (obs_mom.lower[0, 0], obs_mom.upper[0, 0])
            n_bins = 50
            bin_width = (fit_range[1] - fit_range[0]) / n_bins
            
            # --- Main plot ---
            
            mom_plot = np.linspace(fit_range[0], fit_range[1], 200).reshape(-1, 1)

            gauss_curve = zfit.run(gauss.pdf(mom_plot) * result.params[N_RPC]['value'] * bin_width)
            ax1.plot(mom_plot.flatten(), gauss_curve.flatten(), color=colors[i], linestyle="--", label=str(labels[i])+' Fitted Gaussian')
            ax1.grid(True)
            ax1.set_yscale('log')
            data_hist, data_bins, _ = ax1.hist(mom_np, color=colors[i], bins=n_bins, range=fit_range, label=labels[i], histtype='step')
            true_hist, true_bins, _ = ax1.hist(true_rpc, color="orange", bins=n_bins, range=fit_range, label="RPC", histtype='bar')
            data_bin_center = (data_bins[:-1] + data_bins[1:]) / 2
            ax1.errorbar(data_bin_center, data_hist, yerr=np.sqrt(data_hist), fmt='.', color=colors[i], capsize=2)
            
            ax1.set_xlabel('Reconstructed Momentum [MeV/c]')
            ax1.set_ylabel('# of events per bin')
            ax1.legend()
            ax1.set_title('Gaussian Fit to Momentum Data (Extended Unbinned)')
            
            # --- Add text box with fit parameters ---
            param_text = (
                f"Fit parameters for {labels[i]}:\n"
                f"$N_{{RPC}} = {result.params[N_RPC]['value']:.0f} \\pm {hesse_errors[N_RPC]['error']:.2f}$"
            )
            
            props = dict(boxstyle='round', facecolor=colors[i], alpha=0.3)
            
            # Position the text box in the upper left corner of the subplot
            # with an offset for each iteration
            ax1.text(0.4, text_y_pos[i], param_text, transform=ax1.transAxes,
                     fontsize=10, verticalalignment='top', bbox=props)
            
            # --- Ratio plot ---
            
            data_bin_center_2d = data_bin_center.reshape(-1, 1)
            fit_at_bin_center = zfit.run(gauss.pdf(data_bin_center_2d) * result.params[N_RPC]['value'] * bin_width)
            ratio = true_hist / fit_at_bin_center
            
            ax2.errorbar(data_bin_center, ratio, yerr=np.sqrt(data_hist) / fit_at_bin_center, fmt='.', color=colors[i], capsize=2)
            ax2.axhline(1, color='gray', linestyle='--')
            ax2.set_ylabel('Ratio (RPC/Fit)')
            ax2.set_xlabel('Reconstructed Momentum [MeV/c]')
            ax2.set_ylim(0.5, 1.5)
            ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig("RPCfit.pdf")
        plt.show()
