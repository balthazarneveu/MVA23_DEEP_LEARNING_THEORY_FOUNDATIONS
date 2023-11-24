import matplotlib.pyplot as plt
import torch
import numpy as np
from IPython import display
from typing import List
from pathlib import Path


def save_animation(path_list: List[str], output_path: Path):
    try:
        from moviepy.editor import ImageSequenceClip
    except Exception as exc:
        print(f"{exc}\nCannot save GIF, use:\n!pip install moviepy")
        return None
    # Create clip from still debug images
    clip = ImageSequenceClip(path_list, fps=10)
    # Write a gif to disk
    clip.write_gif(str(output_path))


# Useful plot function
def plot_decision_boundary(model, X, Y, epoch, accuracy, model_type='classic',
                           nsamples=100, posterior=None, tloc=(-4, -7),
                           nbh=2, cmap='RdBu', ax=None, force_display=False,
                           title='Classification Analysis'):
    """ Plot and show learning process in classification """
    if ax is None:
        _fig, ax = plt.subplots(figsize=(7, 7))
    h = 0.02*nbh
    x_min, x_max = X[:, 0].min() - 10*h, X[:, 0].max() + 10*h
    y_min, y_max = X[:, 1].min() - 10*h, X[:, 1].max() + 10*h
    xx, yy = np.meshgrid(np.arange(x_min*2, x_max*2, h),
                         np.arange(y_min*2, y_max*2, h))

    test_tensor = torch.from_numpy(
        np.c_[xx.ravel(), yy.ravel()]).type(torch.FloatTensor)
    model.eval()
    with torch.no_grad():
        if model_type == 'classic':
            pred = model(test_tensor)
        elif model_type == 'laplace':
            # Save original mean weight
            original_weight = model.state_dict()['fc.weight'].detach().clone()
            outputs = torch.zeros(nsamples, test_tensor.shape[0], 1)
            for i in range(nsamples):
                state_dict = model.state_dict()
                state_dict['fc.weight'] = torch.from_numpy(
                    posterior[i].reshape(1, 2))
                model.load_state_dict(state_dict)
                outputs[i] = model(test_tensor)

            pred = outputs.mean(0).squeeze()
            state_dict['fc.weight'] = original_weight
            model.load_state_dict(state_dict)

        elif model_type == 'vi':
            outputs = torch.zeros(nsamples, test_tensor.shape[0], 1)
            for i in range(nsamples):
                outputs[i] = model(test_tensor)

            pred = outputs.mean(0).squeeze()

        elif model_type == 'mcdropout':
            # model.eval()
            # model.training = True
            model.train()
            outputs = torch.zeros(nsamples, test_tensor.shape[0], 1)
            for i in range(nsamples):
                outputs[i] = model(test_tensor)
            print(f"AVERAGE OVER {nsamples} {outputs.shape}")
            pred = outputs.mean(0).squeeze()

    Z = pred.reshape(xx.shape).detach().numpy()

    plt.cla()
    ax.set_title(title)
    ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.25)
    ax.contour(xx, yy, Z, colors='k', linestyles=':', linewidths=0.7)
    ax.scatter(X[:, 0], X[:, 1], c=Y, cmap='Paired_r', edgecolors='k')
    ax.text(tloc[0], tloc[1], f'Epoch = {epoch+1}, Accuracy = {accuracy:.2%}',
            fontdict={'size': 12, 'fontweight': 'bold'})
    if force_display:
        display.display(plt.gcf())
        display.clear_output(wait=True)
# Useful function: plot results for linear / polynomial / kernel regression


def plot_results(X_train, y_train, X_test, y_test, y_pred, std_pred,
                 xmin=-2, xmax=2, ymin=-2, ymax=1, stdmin=0.30, stdmax=0.45):
    """Given a dataset and predictions on test set,
    this function draw 2 subplots:
    - left plot compares train set, ground-truth (test set) and predictions
    - right plot represents the predictive variance over input range

    Args:
      X_train: (array) train inputs, sized [N,]
      y_train: (array) train labels, sized [N, ]
      X_test: (array) test inputs, sized [N,]
      y_test: (array) test labels, sized [N, ]
      y_pred: (array) mean prediction, sized [N, ]
      std_pred: (array) std prediction, sized [N, ]
      xmin: (float) min value for x-axis on left and right plot
      xmax: (float) max value for x-axis on left and right plot
      ymin: (float) min value for y-axis on left plot
      ymax: (float) max value for y-axis on left plot
      stdmin: (float) min value for y-axis on right plot
      stdmax: (float) max value for y-axis on right plot

    Returns:
      None
    """
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.xlim(xmin=xmin, xmax=xmax)
    plt.ylim(ymin=ymin, ymax=ymax)
    plt.plot(X_test, y_test, color='green', linewidth=2,
             label="Ground Truth")
    plt.plot(X_train, y_train, 'o', color='blue', label='Training points')
    plt.plot(X_test, y_pred, color='red', label="BLR Poly")
    plt.fill_between(X_test, y_pred-std_pred, y_pred+std_pred,
                     color='indianred', label='1 std. int.')
    plt.fill_between(X_test, y_pred-std_pred*2, y_pred -
                     std_pred, color='lightcoral')
    plt.fill_between(X_test, y_pred+std_pred*1, y_pred +
                     std_pred*2, color='lightcoral', label='2 std. int.')
    plt.fill_between(X_test, y_pred-std_pred*3, y_pred -
                     std_pred*2, color='mistyrose')
    plt.fill_between(X_test, y_pred+std_pred*2, y_pred +
                     std_pred*3, color='mistyrose', label='3 std. int.')
    plt.legend()

    plt.subplot(122)
    plt.title("Predictive variance along x-axis")
    plt.xlim(xmin=xmin, xmax=xmax)
    plt.ylim(ymin=stdmin, ymax=stdmax)
    plt.plot(X_test, std_pred**2, color='red',
             label="\u03C3² {}".format("Pred"))

    # Get training domain
    training_domain = []
    current_min = sorted(X_train)[0]
    for i, elem in enumerate(sorted(X_train)):
        if elem-sorted(X_train)[i-1] > 1:
            training_domain.append([current_min, sorted(X_train)[i-1]])
            current_min = elem
    training_domain.append([current_min, sorted(X_train)[-1]])

    # Plot domain
    for j, (min_domain, max_domain) in enumerate(training_domain):
        plt.axvspan(min_domain, max_domain, alpha=0.5, color='gray',
                    label="Training area" if j == 0 else '')
    plt.axvline(X_train.mean(), linestyle='--', label="Training barycentre")

    plt.legend()
    plt.show()
