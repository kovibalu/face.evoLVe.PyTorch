from PIL import ImageDraw, ImageFont

COLOR_WHEEL = [
    '#ACECD5',
    '#FFF9AA',
    '#FFD5B8',
    '#FFB9B3',
]
UNKNOWN_COLOR = 'white'


def show_results(img, bounding_boxes, facial_landmarks=[], names=[], font=None):
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
    if font is None:
        font = ImageFont.load_default()

    if names:
        assert len(names) == len(bounding_boxes)

    for b_idx, b in enumerate(bounding_boxes):
        if names:
            name, aux_name = names[b_idx]
            if 'unknown' in name.lower():
                outline_color = UNKNOWN_COLOR
            else:
                outline_color = COLOR_WHEEL[hash(name) % len(COLOR_WHEEL)]
            # Draw name under bounding box.
            draw.text((b[0], b[3] + 5),
                      '{} {}'.format(name, aux_name),
                      font=font)
        else:
            outline_color = UNKNOWN_COLOR

        draw.rectangle([
            (b[0], b[1]), (b[2], b[3])
        ], outline=outline_color, width=2)

    if facial_landmarks:
        assert len(facial_landmarks) == len(bounding_boxes)
        for p in facial_landmarks:
            for i in range(5):
                draw.ellipse([
                    (p[i] - 1.0, p[i + 5] - 1.0),
                    (p[i] + 1.0, p[i + 5] + 1.0)
                ], outline='blue')

    return img_copy


def draw_fps(img, font, fps):
    """
    Draws FPS in right-bottom corner of image, right aligned.
    """
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size
    margin = 5
    fps_text = '{:.2f} FPS'.format(fps)
    text_w, text_h = draw.textsize(fps_text, font)
    # Draw in right-bottom corner of image, right aligned.
    draw.text((img_w - margin - text_w, img_h - margin - text_h), fps_text,
              font=font)
