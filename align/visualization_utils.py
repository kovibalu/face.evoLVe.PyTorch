from PIL import ImageDraw, ImageFont

COLOR_WHEEL = [
    '#ACECD5',
    '#FFF9AA',
    '#FFD5B8',
    '#FFB9B3',
]
UNKNOWN_COLOR = 'white'
FONT_PATH = '/usr/share/fonts/dejavu/DejaVuSans.ttf'


def show_results(img, bounding_boxes, facial_landmarks=[], names=[]):
    """Draw bounding boxes and facial landmarks.
    Arguments:
        img: an instance of PIL.Image.
        bounding_boxes: a float numpy array of shape [n, 5].
        facial_landmarks: a float numpy array of shape [n, 10].
    Returns:
        an instance of PIL.Image.
    """
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    font = ImageFont.FreeTypeFont(font=FONT_PATH, size=32)

    if names:
        assert len(names) == len(bounding_boxes)

    for b_idx, b in enumerate(bounding_boxes):
        if names:
            name = names[b_idx]
            outline_color = COLOR_WHEEL[hash(name) % len(COLOR_WHEEL)]
        else:
            name = 'unknown'
            outline_color = UNKNOWN_COLOR

        draw.rectangle([
            (b[0], b[1]), (b[2], b[3])
        ], outline=outline_color, width=2)
        # Draw name under bounding box.
        draw.text((b[0], b[3] + 5), name, font=font)

    if facial_landmarks:
        assert len(facial_landmarks) == len(bounding_boxes)
        for p in facial_landmarks:
            for i in range(5):
                draw.ellipse([
                    (p[i] - 1.0, p[i + 5] - 1.0),
                    (p[i] + 1.0, p[i + 5] + 1.0)
                ], outline='blue')

    return img_copy
