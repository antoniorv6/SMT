import io
import os
import torch
import numpy as np
import matplotlib
from loguru import logger
from PIL import Image
import matplotlib.pyplot as plt
from colorsys import hsv_to_rgb
from matplotlib.cm import ScalarMappable
from matplotlib.colors import (LinearSegmentedColormap, ListedColormap,
                               Normalize)
import moviepy.video.io.ImageSequenceClip
from PIL import Image, ImageDraw, ImageFont
from rich import progress

class DAN_WeightsVisualizer():
    def __init__(self, frames_path, animation_path) -> None:
        self.frames_path = frames_path
        self.video_path = animation_path
        self.add_custom_cmaps()
        logger.info(f"Set frames path in {self.frames_path}")
        self.check_and_create(self.frames_path)
        logger.info(f"Set animations path in {self.video_path}")
        self.check_and_create(self.video_path)

    def check_and_create(self, path):
        if not os.path.exists(path):
            logger.warning(f"The specified path {self.frames_path} does not exists, created it")
            os.makedirs(path)
    
    def add_color_map_alpha(self, color, name, num=256):
        colors = np.zeros((num, 4))
        colors[:, -1] = np.linspace(0.0, 0.75, num)
        for i in range(len(color)):
            colors[:, i] = color[i] / 255
        map_object = LinearSegmentedColormap.from_list(name='{}_alpha'.format(name), colors=colors)
        plt.register_cmap(cmap=map_object)
    
    def add_red_alpha2(self):
        limit = 128
        total = 256
        per_step = int(total / 2)
        color_1 = np.array([0, 0, 1])
        color_2 = np.array([0.5, 1, 0.5])
        color_3 = np.array([1, 0, 0])
        color = np.zeros((total, 4))
        color[:limit, -1] = np.linspace(0.0, 0.6, limit)
        color[limit:, -1] = 0.6

        color[:per_step, :3] = np.linspace(color_1, color_2, per_step)
        color[per_step:, :3] = np.linspace(color_2, color_3, total-per_step)
        # color[:, :3] = np.linspace(color_1, color_2, total)

        map_object = LinearSegmentedColormap.from_list(name='red_alpha2', colors=color)
        plt.register_cmap(cmap=map_object)
    
    def add_custom_cmaps(self):
        self.add_color_map_alpha([255, 0, 0], "red")
        self.add_red_alpha2()
        transparency_ticks = 50
        colors = [hsv_to_rgb(0, a / transparency_ticks, 1) for a in range(transparency_ticks)]
        cmap = ListedColormap(name="red_alpha_bar", colors=colors)
        plt.register_cmap(cmap=cmap)

    def reset(self, img, ground_truth, attention_weights):
        fig = plt.figure(figsize=(50, 30))
        gs = matplotlib.gridspec.GridSpec(nrows=30, ncols=9)
        plt.axis('off')
        plt.suptitle('Sample test')
        x1 = 5
        x15 = 2
        x2 = 7
        text_size = 20
        windows_size = 100  

        anchor_x, anchor_y = (0.01, 0.99)   

        ax_img = plt.subplot(gs[:22, :x1])
        ax_img.get_yaxis().set_visible(False)
        ax_img.get_xaxis().set_visible(False)
        ax_img.title.set_text('Input document') 

        ax_cbar = plt.subplot(gs[22:24, :x15])
        ax_cbar.get_yaxis().set_visible(False)
        ax_cbar.get_xaxis().set_visible(False)
        ax_cbar.axis('off') 

        ax_gt = plt.subplot(gs[:22, x1:x2]) # modified for self-attention
        ax_gt.get_yaxis().set_visible(False)
        ax_gt.get_xaxis().set_visible(False)
        gt_text_0 = ax_gt.text(anchor_x, anchor_y, ground_truth, family='monospace', size=text_size, va="top")
        #ax_gt.text(0.51, 0.99, gt_list)
        ax_gt.title.set_text('Ground truth')
    #
        ax_pred = plt.subplot(gs[:22, 7:])
        ax_pred.get_yaxis().set_visible(False)
        ax_pred.get_xaxis().set_visible(False)
        self.pred_plot = ax_pred.text(anchor_x, anchor_y, "", size=text_size, family='monospace', va='top')
        ax_pred.title.set_text('Prediction')

        #self.ax_auto_attn = plt.subplot(gs[22:, 2:])
        #self.ax_auto_attn.get_yaxis().set_visible(False)
        #self.ax_auto_attn.get_xaxis().set_visible(False)
#
        #self.ax_auto_attn_2 = plt.subplot(gs[24:26, 2:])
        #self.ax_auto_attn_2.get_yaxis().set_visible(False)
        #self.ax_auto_attn_2.get_xaxis().set_visible(False)
#
        #self.ax_auto_attn_3 = plt.subplot(gs[26:, 2:])
        #self.ax_auto_attn_3.get_yaxis().set_visible(False)
        #self.ax_auto_attn_3.get_xaxis().set_visible(False)

        img_input = ax_img.imshow(img, cmap='gray') 

        vmax = np.ceil(np.max(attention_weights)*10)/10
        vmax = 0.1
        norm = Normalize(vmin=0, vmax=vmax)
        self.img_weight = ax_img.imshow(attention_weights[0], cmap="red_alpha", alpha=1, extent=img_input.get_extent(), norm=norm)
        colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap="red_alpha_bar"), ax=ax_cbar, orientation="horizontal", drawedges=False, fraction=1, shrink=0.5, anchor=(0.5, 1.0), ticks=[0, vmax],pad=0.01)
        colorbar.set_label("Attention weights")
        colorbar.ax.set_xticklabels(['0', '{:.1f}+'.format(vmax)])
        plt.subplots_adjust(wspace=0.1, hspace=1, left=0.01, right=0.99, top=0.93, bottom=0.01) 
    
    def generate_typed_text_line_image_attention(self, text, prt="", bg_color=(255,255,255), txt_color=(0,0,0), color_mode="RGB", padding=(5,5), font_size=9):
        """
        Create an image for each character in text then concatenate the images.
        """
        font_path = "Fonts/DejaVuSansMono-Bold.ttf"
        fnt = ImageFont.truetype(font_path, font_size)
        padding_top, padding_bottom = padding
        _, text_height = fnt.getsize(text)
        img_height = padding_top + padding_bottom + text_height
        imgs = []

        # compute the maximum character width
        max_width = max([fnt.getsize(c)[0] for c in text.split(" ")])

        for c in text:
            c_width, _ = fnt.getsize(c)
            padding_width = ((max_width + 10) - c_width) // 2 # add a margin of 10 pixels
            img_width = padding_width * 2 + c_width
            img = Image.new(color_mode, (img_width, img_height), color=bg_color)
            d = ImageDraw.Draw(img)
            d.text((padding_width, padding_bottom), c, font=fnt, fill=txt_color, spacing=0)
            imgs.append(np.array(img))


        return np.concatenate(imgs,axis=1)

    def render(self, x, y, predicted_seq, attn_weights, self_weights, animation_name="def", windows_size=100):
        framespath = f"{self.frames_path}{animation_name}"
        os.makedirs(f"{framespath}", exist_ok=True)

        self.reset(img=x, ground_truth=y, attention_weights=attn_weights)
        frame_buffer = []
        
        #vmax_self = np.ceil(np.max(np.array(self_weights[-1].cpu()))*10)/10 # round to the superior 0.1 value
        #norm_self = Normalize(vmin=0, vmax=vmax_self)
        #ax = self.ax_auto_attn

        for idx in progress.track(range(len(predicted_seq))):
            # PLOT NEW ATTENTION WEIGHTS ON IMAGE
            self.img_weight.set_data(attn_weights[idx])
            
            # PLOT PREDICTION TEXT (WOULD BE COOL TO HIGHLIGHT ERRORS)
            text_to_plot = "".join(predicted_seq[:idx]).replace("<b>", "\t¶\n").replace("<t>", "\t--\t").replace("<s>", "\t-\t").expandtabs(2)
            self.pred_plot.set_text(text_to_plot)
            
            # PLOT SELF-ATTENTION WEIGHTS
            #rectangles = []
            #if idx > 1:
            #    plot_attention = self_weights[idx-1]
            #    plot_attention = torch.squeeze(plot_attention)
            #    if bool(plot_attention.shape):
            #        plot_attention = plot_attention.tolist()
            #    else:
            #        plot_attention = [plot_attention]
            #    
            #    chars_used = predicted_seq[:len(plot_attention)-1]
            #    if len(chars_used) > 100:
            #        chars_used = predicted_seq[-100:]
            #
            #    # Fixed height and width for each token
            #    token_height = 0.05  # Adjust as needed
            #    token_width = 0.05   # Adjust as needed
#
            #    # Calculate the number of tokens that can fit in one row
            #    tokens_per_row = int(1 / token_width)
            #    
            #    ax = self.ax_auto_attn
            #    alpha_values = [norm_self(alpha) for alpha in plot_attention[idx-1]]
            #    if len(alpha_values) > 100:
            #        alpha_values = alpha_values[-100:]
            #    
            #    for i, word in enumerate(chars_used):
            #        if word == '<b>':
            #            word = '¶'
            #        if word == '<s>':
            #            word = '-'
            #        if word == '<t>':
            #            word = '--'
#
            #        # Calculate the row and column index for token placement
            #        row = i // tokens_per_row
            #        col = i % tokens_per_row
#
            #        # Calculate the x and y coordinates based on token width and height
            #        x = col * token_width
            #        y = 1 - (row + 1) * token_height
#
            #        # Create a rectangle with fixed height and width
            #        if alpha_values[i]>0.02:
            #            #print(alpha_values)
            #            rect = plt.Rectangle((x, y), token_width, token_height, fill=True, facecolor='r', alpha=alpha_values[i])
            #            rectangles.append(rect)
            #            # Add the word as text in the center of the rectangle
            #            ax.add_patch(rect)
            #        ax.text(x + token_width / 2, y + token_height / 2, word, ha='center', va='center', fontsize=12)
            
            plt.savefig(f'{framespath}/{idx}.png', bbox_inches='tight', pad_inches=0)
            frame_buffer.append(f'{framespath}/{idx}.png')
            
            #for rectangle in rectangles:
            #    rectangle.remove()
            #
            #rectangles = []

        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frame_buffer, fps=2)
        clip.write_videofile(f'{self.video_path}/{animation_name}.mp4')


    

