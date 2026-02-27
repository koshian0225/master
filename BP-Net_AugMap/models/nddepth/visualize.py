import numpy as np
import matplotlib.pyplot as plt

def visualize_image(image,ax=None):
    if ax is None:
        fig, ax = plt.subplots(1)
    ax.set_xlim(0,image.shape[1])
    ax.set_ylim(0,image.shape[0])  
    ax.set_axis_off()
    ax.imshow(image)   
    ax.invert_yaxis()
    if ax is None:
        plt.style.use('dark_background')
        plt.show()  
    return ax,image

def visualize_depth(depth,ax=None):
    if ax is None:
        fig, ax = plt.subplots(1)
    ax.set_xlim(0,depth.shape[1])
    ax.set_ylim(0,depth.shape[0])  
    ax.set_axis_off()
    ax.imshow(depth,cmap='jet')   
    ax.invert_yaxis()
    if ax is None:
        plt.style.use('dark_background')
        plt.show()  
    return ax,depth

def visualize_depth_alphamask(depth,mask:np.ndarray,ax=None):
    '''
    mask : np.ndarray
    [[True,False]]の二次元配列
    Trueの場合色を表示させる
    '''
    if ax is None:
        fig, ax = plt.subplots(1)
    ax.set_xlim(0,depth.shape[1])
    ax.set_ylim(0,depth.shape[0])  
    ax.set_axis_off()
    # alpha_mask = np.ones_like(depth, dtype=float)
    # alpha_mask[mask] = 0  # Trueの領域を透明にする
    ax.imshow(mask,alpha=0.3,cmap='gray')   
    ax.imshow(depth,cmap='jet',alpha=0.7)   
    ax.invert_yaxis()
    if ax is None:
        plt.style.use('dark_background')
        plt.show()  
    return ax,depth

def visualize_label_alphamask(label,mask:np.ndarray,ax=None):
    if ax is None:
        fig, ax = plt.subplots(1)
    ax.set_xlim(0,label.shape[1])
    ax.set_ylim(0,label.shape[0])  
    ax.set_axis_off()
    ax.imshow(mask,alpha=0.3,cmap='gray')   
    ax.imshow(label,alpha=0.7)   
    ax.invert_yaxis()
    if ax is None:
        plt.style.use('dark_background')
        plt.show()  
    return ax,label


def visualize_label(label,ax=None):
    if ax is None:
        fig, ax = plt.subplots(1)
    ax.set_xlim(0,label.shape[1])
    ax.set_ylim(0,label.shape[0])  
    ax.set_axis_off()
    ax.imshow(label)   
    ax.invert_yaxis()
    if ax is None:
        plt.style.use('dark_background')
        plt.show()  
    return ax,label

