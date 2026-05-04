print("Script baslatiliyor...")
import os
import json
import cv2 
import numpy as np
from typing import Optional

"""
s - save frame
q - exit
d - prev frame
f - next frame
a - first frame
k - prev video
l - next video 
j - first video
h - go to video
space - play/stop
i - go to frame
r - rotate 90 deg
"""

def resize_with_pad(image: np.ndarray, target_size: tuple) -> np.ndarray:
    """Resizes image to target size maintaining aspect ratio and padding with black."""
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create black canvas
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Calculate offset
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    # Paste
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

def get_input_overlay(img, title: str, prompt: str, min_val: int, max_val: int) -> Optional[int]:
    """Custom input dialog overlaid on main image"""
    input_str = ""
    original_img = img.copy()
    
    while True:
        # Start with original image
        display_img = original_img.copy()
        
        # Create semi-transparent overlay
        overlay = display_img.copy()
        cv2.rectangle(overlay, (400, 300), (1200, 600), (0, 0, 0), -1)
        cv2.addWeighted(display_img, 0.3, overlay, 0.7, 0, display_img)
        
        # Draw input dialog on overlay
        cv2.rectangle(display_img, (420, 320), (1180, 580), (100, 100, 100), 3)
        
        # Draw title and prompt
        cv2.putText(display_img, title, (450, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(display_img, prompt, (450, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(display_img, f"Range: {min_val}-{max_val}", (450, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        
        # Draw input box
        cv2.rectangle(display_img, (450, 450), (1150, 500), (100, 100, 100), 2)
        cv2.rectangle(display_img, (452, 452), (1148, 498), (50, 50, 50), -1)
        cv2.putText(display_img, input_str, (460, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw instructions
        cv2.putText(display_img, "Enter number, then press ENTER", (450, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(display_img, "Press ESC to cancel", (450, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        cv2.imshow('image', display_img)
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == 27:  
            return None
        elif key == 13:  
            try:
                value = int(input_str)
                if min_val <= value <= max_val:
                    return value
                else:
                    return None
            except ValueError:
                return None
        elif key == 8:  
            input_str = input_str[:-1]
        elif key >= 48 and key <= 57:  
            input_str += chr(key)


def rotate_image(image: np.ndarray, steps: int) -> np.ndarray:
    """Rotate image by 90-degree steps clockwise."""
    steps = steps % 4
    if steps == 1:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if steps == 2:
        return cv2.rotate(image, cv2.ROTATE_180)
    if steps == 3:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

app_state = {
    'rotation_steps': 0,
    'show_help': False,
    'zoom': 1.0,
    'pan_x': 0,
    'pan_y': 0,
    'dragging': False,
    'last_mouse': (0, 0),
    'speed': 1.0,
    'play': True
}

def apply_zoom_pan(image):
    """Crops the image based on zoom level and pan offset."""
    zoom = app_state['zoom']
    if zoom <= 1.0:
        return image
    
    h, w = image.shape[:2]
    
   
    vw, vh = int(w / zoom), int(h / zoom)
    
    
    cx = w // 2 + app_state['pan_x']
    cy = h // 2 + app_state['pan_y']
    
    
    cx = max(vw // 2, min(cx, w - vw // 2))
    cy = max(vh // 2, min(cy, h - vh // 2))
    
    
    x1 = cx - vw // 2
    y1 = cy - vh // 2
    
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x1 + vw), min(h, y1 + vh)
    
    return image[y1:y2, x1:x2]

def draw_help_overlay(img):
    """Draws a semi-transparent help overlay with keyboard shortcuts."""
    overlay = img.copy()
    h, w = img.shape[:2]
    
    
    cv2.rectangle(overlay, (200, 100), (w-200, h-100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    
    
    cv2.rectangle(img, (200, 100), (w-200, h-100), (255, 255, 255), 2)
    
    # Title
    cv2.putText(img, "KEYBOARD SHORTCUTS", (w//2 - 150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    
    shortcuts = [
        "SPACE : Play / Stop",
        "L / K : Next / Prev Video",
        "F / D : Next / Prev Frame",
        "A     : First Frame",
        "I     : Go to Frame",
        "H     : Toggle Video Selection",
        "R     : Rotate 90 deg",
        "S     : Save Frame",
        "+ / - : Speed Up/Down",
        "*     : Reset Speed (1.0x)",
        "Wheel : Zoom In/Out",
        "Drag  : Pan Image",
        "?     : Toggle This Help",
        "Q     : Quit"
    ]
    
    start_y = 220
    for i, line in enumerate(shortcuts):
        cv2.putText(img, line, (250, start_y + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

def on_mouse(event, x, y, flags, param):
    global app_state
    
    if event == cv2.EVENT_LBUTTONDOWN:
        
        if 1240 <= x <= 1350 and 25 <= y <= 90: # Left Button
            app_state['rotation_steps'] = (app_state['rotation_steps'] - 1) % 4
        elif 1360 <= x <= 1470 and 25 <= y <= 90: # Right Button
            app_state['rotation_steps'] = (app_state['rotation_steps'] + 1) % 4
        else:
            
            app_state['dragging'] = True
            app_state['last_mouse'] = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if app_state['dragging'] and app_state['zoom'] > 1.0:
            dx = x - app_state['last_mouse'][0]
            dy = y - app_state['last_mouse'][1]
            
            
            speed = 2
            app_state['pan_x'] -= dx * speed
            app_state['pan_y'] -= dy * speed
            
            app_state['last_mouse'] = (x, y)
            
    elif event == cv2.EVENT_LBUTTONUP:
        app_state['dragging'] = False
        
    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            app_state['zoom'] = min(app_state['zoom'] + 0.1, 5.0)
        else:
            app_state['zoom'] = max(app_state['zoom'] - 0.1, 1.0)
            if app_state['zoom'] == 1.0:
                app_state['pan_x'] = 0
                app_state['pan_y'] = 0




path_root = os.getcwd()
path = path_root
save_path = os.path.join(path_root, 'images')
resolution = (1600, 900)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('image', on_mouse)

try:
    all_listdir = sorted([f for f in os.listdir(path) if f.lower().endswith('.mp4')])
    listdir = [os.path.splitext(_)[0] for _ in all_listdir]
    print(f"Bulunan videolar: {all_listdir}")
    listdir = sorted(list(set(listdir)))
except Exception as e:
    print(f"Hata: {e}")
    listdir = []

imgList=[]

if os.path.exists(save_path):
    img_Listdir = sorted(os.listdir(save_path))
    imgList = [os.path.splitext(_)[0] for _ in img_Listdir]
    imgList = set(imgList)
else:
    if os.path.exists(path_root):
        os.makedirs(save_path)

imgTotal=len(imgList)
imgCount=0

i = 0
i_while = True
if not listdir:
    i_while = False

while i_while:
    name = listdir[i]
    name_path = os.path.join(path, name)

    cap = cv2.VideoCapture(name_path + '.mp4')
    if not cap.isOpened():
        i += 1
        if i >= len(listdir):
            break
        continue

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    description = None
    try:
        with open(name_path + '.json', encoding='utf-8') as f:
            description = json.loads(f.read())
    except: 
        print("json okunamadi")
 
    print('*' * 30)
    print(name)
    if description!=None:
        print(description['text'])

    i_space = 1  # Otomatik oynatmayı aktif et (0: duraklatılmış, 1: oynuyor)

    j = 0
    j_while = True
    current_j = -1
    while j_while:
        if j != current_j + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, j)
        
        ret, img = cap.read()
        if not ret:
            if j >= num_frames - 1:
                 print("Video sonuna gelindi.")
                 j_while = False
                 continue
            print(f'\nKare okunamadi: {j}, tekrar deneniyor...')
            j += 1
            continue
        
        current_j = j
        imgSave = img.copy()
        #döndürme işlemi
        rotation = app_state['rotation_steps']
        if rotation:
            imgSave = rotate_image(imgSave, rotation)
            img = rotate_image(img, rotation)
        
        # Apply Zoom & Pan
        img = apply_zoom_pan(img)
            
        img = resize_with_pad(img, resolution)
        
        # Add UI overlays
        img=cv2.rectangle(img,(1480,25),(1590,90),(0,0,0),-1)
        img=cv2.rectangle(img,(23,2),(150,25),(0,0,0),-1)
        img = cv2.putText(img, f'{i+1}/{len(listdir)}', (1490, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),1)
        img = cv2.putText(img, f'{imgTotal} + {imgCount}', (1490, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),1)
        img = cv2.putText(img, f'{j+1}/{num_frames}', (25, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),1)
        
    
        
        
        cv2.rectangle(img, (1240, 25), (1350, 90), (60, 60, 60), -1) #sola döndürme butonu
        cv2.rectangle(img, (1240, 25), (1350, 90), (200, 200, 200), 2)
        cv2.putText(img, "<< L", (1255, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        
        cv2.rectangle(img, (1360, 25), (1470, 90), (60, 60, 60), -1) #sağa döndürme butonu
        cv2.rectangle(img, (1360, 25), (1470, 90), (200, 200, 200), 2)
        cv2.putText(img, "R >>", (1375, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw Video Name
        cv2.putText(img, f'Video: {name}', (50, 850), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Draw Original Resolution Info - TOP CENTER
        resolution_text = f"Orig: {w}x{h}"
        if app_state.get('zoom', 1.0) > 1.0:
            resolution_text += f" | Zoom: {app_state['zoom']:.1f}x"
        
        if app_state['speed'] != 1.0:
            resolution_text += f" | Spd: {app_state['speed']:.1f}x"
        
        text_size = cv2.getTextSize(resolution_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_w = text_size[0]
        center_x = 1600 // 2
        
        res_box_x1 = center_x - (text_w // 2) - 10
        res_box_x2 = center_x + (text_w // 2) + 10
        
        cv2.rectangle(img, (res_box_x1, 25), (res_box_x2, 65), (0, 0, 0), -1)
        cv2.putText(img, resolution_text, (res_box_x1 + 10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Draw Help Overlay if active
        if app_state['show_help']:
            draw_help_overlay(img)
        else:
            # Draw tiny help hint if help is closed
            cv2.putText(img, "Press '?' for Help", (w - 300, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        cv2.imshow('image', img)
 
        delay = int(33 / app_state['speed'])
        key = cv2.waitKey(max(1, delay)) & 0xFF
        if key == ord('q'):
            cap.release()
            i_while = False
            j_while = False
        elif key == ord('?'): # Toggle Help
            app_state['show_help'] = not app_state['show_help']
        elif key == ord('d'):
            j = j - 1 if j > 0 else 0
        elif key == ord('f'):
            j = j + 1 if j < num_frames - 1 else num_frames - 1
        elif key == ord('k'):
            i = i - 1 if i > 0 else 0
            if i != 0:
                cap.release()
            j_while = False
        elif key == ord('l'):
            i = i + 1 if i < len(listdir) - 1 else len(listdir) - 1
            if i != len(listdir) - 1:
                cap.release()
            j_while = False
        elif key == ord('j'):
            i = 0
            cap.release()
            j_while = False
        elif key == ord('a'):
            j=0
        elif key == ord('i'):
            go_to_input = get_input_overlay(img, "Go to Frame", f"Frame (1-{num_frames}):", 1, num_frames)
            if go_to_input is not None:
                go_to = go_to_input - 1
                j = go_to
        elif key == ord('h'):
            goVideo_input = get_input_overlay(img, "Go to Video", f"Video (1-{len(listdir)}):", 1, len(listdir))
            if goVideo_input is not None:
                goVideo = goVideo_input - 1
                i = goVideo
                j_while=False
        elif key == ord('s'):
            suffix_part = f"{name}_{j}.jpg"

            already_exists = any(
                file.endswith('.jpg') and file.split('_', 1)[-1] == suffix_part
                for file in os.listdir(save_path)
            )

            if not already_exists:
                next_index = max(
                    [int(f[:5]) for f in os.listdir(save_path) if f[:5].isdigit()] + [-1]
                ) + 1
                save_name = os.path.join(save_path, f"{next_index:05d}_{suffix_part}")
                deger = cv2.imwrite(save_name, imgSave)
                print('Saved:', save_name)
                imgCount += 1
            else:
                print('Already saved:', suffix_part)

        elif key == ord(' '):
            i_space = int(not i_space)
        elif key == ord('r'):
            app_state['rotation_steps'] = (app_state['rotation_steps'] + 1) % 4
        elif key == ord('+') or key == 43:
            app_state['speed'] = min(app_state['speed'] + 0.25, 5.0)
        elif key == ord('-') or key == 45:
            app_state['speed'] = max(app_state['speed'] - 0.25, 0.25)
        elif key == ord('*') or key == 42:
            app_state['speed'] = 1.0

        if i_space:
            if j < num_frames - 1:
                j += 1
            else:
                j = num_frames - 1
                i_space = 0
                print("Video durduruldu (son kare).")
        
            
cv2.destroyAllWindows()
