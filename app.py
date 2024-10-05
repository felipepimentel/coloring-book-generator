import argparse
import os
import cv2
import numpy as np
from PIL import Image

def create_coloring_page(input_path, output_path, args):
    # Ler a imagem com OpenCV
    img = cv2.imread(input_path)
    if img is None:
        print(f"Erro ao ler a imagem: {input_path}")
        return

    # Converter para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar filtro bilateral para reduzir ruído e preservar bordas
    blurred = cv2.bilateralFilter(
        gray,
        d=args.bilateral_d,
        sigmaColor=args.bilateral_sigma_color,
        sigmaSpace=args.bilateral_sigma_space
    )

    # Escolher o método de detecção de bordas
    if args.method == 'canny':
        edges = cv2.Canny(blurred, args.canny_threshold1, args.canny_threshold2)
    elif args.method == 'laplacian':
        edges = cv2.Laplacian(blurred, cv2.CV_8U, ksize=5)
    elif args.method == 'sobel':
        sobelx = cv2.Sobel(blurred, cv2.CV_8U, 1, 0, ksize=5)
        sobely = cv2.Sobel(blurred, cv2.CV_8U, 0, 1, ksize=5)
        edges = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    elif args.method == 'sketch':
        inverted_gray = 255 - blurred
        blurred_inverted = cv2.GaussianBlur(inverted_gray, (21, 21), 0)
        edges = cv2.divide(gray, 255 - blurred_inverted, scale=256)
    else:
        print(f"Método desconhecido: {args.method}")
        return

    # Normalizar e converter para 8 bits se necessário
    if edges.dtype != np.uint8:
        edges = cv2.normalize(edges, edges, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Aplicar limiarização
    if args.adaptive_threshold:
        thresh = cv2.adaptiveThreshold(
            edges,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
    else:
        _, thresh = cv2.threshold(edges, args.threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Remover pequenos ruídos
    kernel_size = args.morph_kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=args.morph_iterations)

    # Inverter cores para ter linhas pretas em fundo branco
    result = cv2.bitwise_not(cleaned)

    # Redimensionar para o tamanho desejado
    dpi = args.dpi
    width_in_px = int(args.width_cm / 2.54 * dpi)
    height_in_px = int(args.height_cm / 2.54 * dpi)
    size = (width_in_px, height_in_px)

    pil_img = Image.fromarray(result)

    # Redimensionar mantendo a proporção
    pil_img.thumbnail(size, Image.LANCZOS)

    # Criar fundo branco e centralizar a imagem
    background = Image.new('L', size, 255)
    position = ((size[0] - pil_img.width) // 2, (size[1] - pil_img.height) // 2)
    background.paste(pil_img, position)

    # Salvar a imagem resultante
    background.save(output_path, dpi=(dpi, dpi))

def main():
    parser = argparse.ArgumentParser(description='Converter imagens para páginas de livro de colorir de alta qualidade.')
    parser.add_argument('images', nargs='+', help='Caminhos das imagens de entrada')
    parser.add_argument('-o', '--output', default='output', help='Diretório de saída')
    parser.add_argument('-m', '--method', choices=['canny', 'laplacian', 'sobel', 'sketch'], default='canny', help='Método de detecção de bordas')
    parser.add_argument('--canny-threshold1', type=int, default=50, help='Primeiro limiar para o detector de bordas Canny')
    parser.add_argument('--canny-threshold2', type=int, default=150, help='Segundo limiar para o detector de bordas Canny')
    parser.add_argument('--threshold-value', type=int, default=127, help='Valor de limiar para binarização')
    parser.add_argument('--adaptive-threshold', action='store_true', help='Usar limiarização adaptativa')
    parser.add_argument('--width-cm', type=float, default=21.0, help='Largura da imagem em centímetros')
    parser.add_argument('--height-cm', type=float, default=29.7, help='Altura da imagem em centímetros')
    parser.add_argument('--dpi', type=int, default=300, help='Resolução da imagem em DPI')
    parser.add_argument('--bilateral-d', type=int, default=9, help='Diâmetro do pixel para o filtro bilateral')
    parser.add_argument('--bilateral-sigma-color', type=float, default=75, help='Sigma Color para o filtro bilateral')
    parser.add_argument('--bilateral-sigma-space', type=float, default=75, help='Sigma Space para o filtro bilateral')
    parser.add_argument('--morph-kernel-size', type=int, default=3, help='Tamanho do kernel para operações morfológicas')
    parser.add_argument('--morph-iterations', type=int, default=1, help='Número de iterações para operações morfológicas')

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    for image_path in args.images:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(args.output, f"{base_name}_coloring.png")
        create_coloring_page(image_path, output_path, args)
        print(f"Página de colorir criada: {output_path}")

if __name__ == "__main__":
    main()
