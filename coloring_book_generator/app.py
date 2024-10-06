import tkinter as tk
from tkinter import filedialog

import customtkinter as ctk
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageTk
from torchvision import transforms

from .model import U2NET  # Revert to original model


def load_model(model_path):
    model = U2NET(3, 1)
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True), strict=False
    )
    model.eval()
    return model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def process_image(
    model,
    input_image_path,
    threshold_value,
    canny_min,
    canny_max,
    dilation_iter,
    erosion_iter,
    gaussian_blur,
):
    input_image = Image.open(input_image_path).convert("RGB")
    input_tensor = preprocess_image(input_image)

    with torch.no_grad():
        prediction = model(input_tensor)
        pred = prediction[0][:, 0, :, :]

    coloring_book_image = postprocess_prediction(
        pred,
        input_image,
        threshold_value,
        canny_min,
        canny_max,
        dilation_iter,
        erosion_iter,
        gaussian_blur,
    )
    return coloring_book_image, input_image


def preprocess_image(image):
    transform = transforms.Compose(
        [
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return (
        transform(image)
        .unsqueeze(0)
        .to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    )


def postprocess_prediction(
    pred,
    original_image,
    threshold_value,
    canny_min,
    canny_max,
    dilation_iter,
    erosion_iter,
    gaussian_blur,
):
    pred = pred.squeeze().cpu().numpy()
    pred_min = pred.min()
    pred_max = pred.max()
    pred = (pred - pred_min) / (pred_max - pred_min)

    # Convert to binary image
    _, mask = cv2.threshold(
        (pred * 255).astype(np.uint8), threshold_value, 255, cv2.THRESH_BINARY
    )

    # Apply Gaussian blur for smoothing
    mask = cv2.GaussianBlur(mask, (gaussian_blur, gaussian_blur), 0)

    # Apply edge detection to enhance lines
    edges = cv2.Canny(mask, canny_min, canny_max)

    # Optionally apply dilation and erosion to improve line quality
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=dilation_iter)
    edges = cv2.erode(edges, kernel, iterations=erosion_iter)

    # Invert edges for coloring book effect
    edges = 255 - edges

    # Resize to match original image size
    edges = cv2.resize(
        edges,
        (original_image.width, original_image.height),
        interpolation=cv2.INTER_LINEAR,
    )
    return Image.fromarray(edges.astype(np.uint8))


class ColoringBookGenerator:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Coloring Book Generator")
        self.root.geometry("1200x900")
        self.input_image_path = tk.StringVar()
        self.model = load_model("./assets/u2net.pth")  # Updated model path
        self.threshold_value = tk.IntVar(value=127)
        self.canny_min = tk.IntVar(value=50)
        self.canny_max = tk.IntVar(value=100)
        self.dilation_iter = tk.IntVar(value=1)
        self.erosion_iter = tk.IntVar(value=1)
        self.gaussian_blur = tk.IntVar(value=5)
        self.setup_ui()

    def setup_ui(self):
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Create a frame for controls
        controls_frame = ctk.CTkFrame(main_frame)
        controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        select_button = ctk.CTkButton(
            controls_frame, text="Select and Generate", command=self.select_image
        )
        select_button.pack(pady=10)

        # Threshold slider
        threshold_slider = ctk.CTkSlider(
            controls_frame,
            from_=0,
            to=255,
            variable=self.threshold_value,
            command=self.update_preview,
        )
        threshold_slider.pack(pady=10)
        threshold_label = ctk.CTkLabel(controls_frame, text="Threshold Value")
        threshold_label.pack()

        # Canny min slider
        canny_min_slider = ctk.CTkSlider(
            controls_frame,
            from_=0,
            to=255,
            variable=self.canny_min,
            command=self.update_preview,
        )
        canny_min_slider.pack(pady=10)
        canny_min_label = ctk.CTkLabel(controls_frame, text="Canny Min Value")
        canny_min_label.pack()

        # Canny max slider
        canny_max_slider = ctk.CTkSlider(
            controls_frame,
            from_=0,
            to=255,
            variable=self.canny_max,
            command=self.update_preview,
        )
        canny_max_slider.pack(pady=10)
        canny_max_label = ctk.CTkLabel(controls_frame, text="Canny Max Value")
        canny_max_label.pack()

        # Dilation iterations slider
        dilation_slider = ctk.CTkSlider(
            controls_frame,
            from_=0,
            to=5,
            variable=self.dilation_iter,
            command=self.update_preview,
        )
        dilation_slider.pack(pady=10)
        dilation_label = ctk.CTkLabel(controls_frame, text="Dilation Iterations")
        dilation_label.pack()

        # Erosion iterations slider
        erosion_slider = ctk.CTkSlider(
            controls_frame,
            from_=0,
            to=5,
            variable=self.erosion_iter,
            command=self.update_preview,
        )
        erosion_slider.pack(pady=10)
        erosion_label = ctk.CTkLabel(controls_frame, text="Erosion Iterations")
        erosion_label.pack()

        # Gaussian blur slider
        gaussian_blur_slider = ctk.CTkSlider(
            controls_frame,
            from_=1,
            to=15,
            variable=self.gaussian_blur,
            command=self.update_preview,
        )
        gaussian_blur_slider.pack(pady=10)
        gaussian_blur_label = ctk.CTkLabel(
            controls_frame, text="Gaussian Blur Kernel Size"
        )
        gaussian_blur_label.pack()

        # Preview frame
        self.preview_frame = ctk.CTkFrame(main_frame)
        self.preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        # Manual drawing tools
        self.drawing_mode = tk.StringVar(value="none")
        draw_button = ctk.CTkButton(
            controls_frame, text="Draw", command=lambda: self.set_drawing_mode("draw")
        )
        draw_button.pack(pady=5)
        erase_button = ctk.CTkButton(
            controls_frame, text="Erase", command=lambda: self.set_drawing_mode("erase")
        )
        erase_button.pack(pady=5)
        self.canvas = None

    def set_drawing_mode(self, mode):
        self.drawing_mode.set(mode)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if file_path:
            self.input_image_path.set(file_path)
            self.update_preview()

    def update_preview(self, *args):
        if not self.input_image_path.get():
            return

        try:
            processed, original = process_image(
                self.model,
                self.input_image_path.get(),
                self.threshold_value.get(),
                self.canny_min.get(),
                self.canny_max.get(),
                self.dilation_iter.get(),
                self.erosion_iter.get(),
                self.gaussian_blur.get(),
            )
            self.display_preview(processed, original)
        except Exception as e:
            self.show_error_dialog(str(e))

    def display_preview(self, processed, original):
        for widget in self.preview_frame.winfo_children():
            widget.destroy()

        # Display original image
        original.thumbnail((300, 300), Image.LANCZOS)
        original_tk = ImageTk.PhotoImage(original)
        original_label = tk.Label(
            self.preview_frame, image=original_tk, text="Original Image"
        )
        original_label.image = original_tk
        original_label.pack(side=tk.LEFT, padx=10, pady=10)

        # Display processed image
        processed.thumbnail((300, 300), Image.LANCZOS)
        processed_tk = ImageTk.PhotoImage(processed)
        processed_label = tk.Label(
            self.preview_frame, image=processed_tk, text="Processed Image"
        )
        processed_label.image = processed_tk
        processed_label.pack(side=tk.RIGHT, padx=10, pady=10)

        # Enable manual editing on processed image
        self.canvas = tk.Canvas(
            self.preview_frame, width=processed.width, height=processed.height
        )
        self.canvas.pack()
        self.processed_image = processed.copy()
        self.tk_image = ImageTk.PhotoImage(self.processed_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.bind("<B1-Motion>", self.manual_draw)

    def manual_draw(self, event):
        if self.drawing_mode.get() == "none":
            return

        x, y = event.x, event.y
        draw = ImageDraw.Draw(self.processed_image)
        color = 0 if self.drawing_mode.get() == "erase" else 255
        draw.rectangle([x - 2, y - 2, x + 2, y + 2], fill=color)

        self.tk_image = ImageTk.PhotoImage(self.processed_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def show_error_dialog(self, error_message):
        error_window = ctk.CTkToplevel(self.root)
        error_window.title("Error")
        error_window.geometry("400x200")

        error_label = ctk.CTkLabel(
            error_window, text="An error occurred:", font=("Arial", 16, "bold")
        )
        error_label.pack(pady=(20, 10))

        error_text = ctk.CTkTextbox(error_window, height=100, width=360)
        error_text.insert("1.0", error_message)
        error_text.configure(state="disabled")
        error_text.pack(padx=20, pady=10)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
    app = ColoringBookGenerator()
    app.run()
