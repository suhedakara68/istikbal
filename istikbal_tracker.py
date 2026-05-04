import cv2
import numpy as np
from ultralytics import YOLO
import json
import os
from PIL import Image, ImageDraw, ImageFont

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- CONFIG ---
MODEL_PATH = "best.pt"
VIDEO_PATH = "istikball.mp4"

# Sınıf ID'leri (yeni model)
CLS_INSAN = 0
CLS_MAKINE = 1
CLS_PARCA = 2

# Eşikler  
MACHINE_IDLE_TIMEOUT = 3.0     # Makine durup bu kadar sn beklerse biter
MACHINE_MOTION_THRESHOLD = 1.5 # Titreşimleri engellemek için eşik yükseltildi
MACHINE_HOME_RATIO = 0.5       
MACHINE_PERSIST_FRAMES = 150   
MIN_CYCLE_DURATION = 1.0       
MIN_CARRY_DURATION = 0.5       # Çok kısa taşıma süreleri filtrelenir
BOX_INTERSECT_TOLERANCE = 50   # İşçi-parça çakışma toleransı (px)
MIN_OVERLAP_THRESHOLD = 0.30   # Minimum overlap oranı (parça taşıyor sayılması için)
CARRY_PERSIST_FRAMES = 60      # Parça kaybedilse bile 2sn boyunca taşıma devam eder
CARRY_COOLDOWN_FRAMES = 90     # Bu süre içinde tekrar parça alırsa sayaç devam eder (sıfırlanmaz)
PLAYBACK_SKIP = 2              # Her N frame'de bir işlem yap (Hızlandırmak için)
PLAYBACK_DELAY = 1             # cv2.waitKey süresi (ms)

# --- YARDIMCI FONKSİYONLAR ---
def get_center(bbox):
    return (int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2))

def calc_iou(b1, b2):
    """İki bbox'ın IoU (Intersection over Union) değerini hesaplar"""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    if inter == 0: return 0.0
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / min(a1, a2)  # Küçük kutuya göre oran

def boxes_intersect(b1, b2, tol=BOX_INTERSECT_TOLERANCE):
    return not (b1[2]+tol < b2[0] or b1[0]-tol > b2[2] or
                b1[3]+tol < b2[1] or b1[1]-tol > b2[3])

def is_inside(point, roi):
    if roi is None or roi.size == 0:
        return False
    return cv2.pointPolygonTest(roi, (float(point[0]), float(point[1])), False) >= 0

# --- MODERN UI ENGINE (PIL tabanlı) ---
def get_font(size=20, bold=False):
    try:
        # Windows UI fontu (iOS/iPhone stiline en yakın temiz font)
        font_path = "C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"
        return ImageFont.truetype(font_path, size)
    except:
        return ImageFont.load_default()

def draw_styled_text(img, text, pos, size=20, color=(255, 255, 255), bold=False):
    # OpenCV (BGR) -> PIL (RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = get_font(size, bold)
    draw.text(pos, text, font=font, fill=color)
    # PIL -> OpenCV (BGR)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_glass_panel(img, x1, y1, x2, y2, color=(30, 30, 40), alpha=0.8):
    sub = img[y1:y2, x1:x2]
    rect = np.zeros(sub.shape, dtype=np.uint8)
    rect[:] = color
    res = cv2.addWeighted(sub, 1-alpha, rect, alpha, 0)
    img[y1:y2, x1:x2] = res
    # İnce parlak kenar (Glow effect)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
    return img

# --- ROI YÜKLEME ---
def load_rois():
    machine_roi, load_roi, unload_roi = None, None, None
    if os.path.exists("rois.json"):
        try:
            with open("rois.json", "r") as f:
                data = json.load(f)
                for item in data:
                    pts = np.array(item["pts"], np.int32)
                    if item.get("id") == 1: machine_roi = pts
                    elif item.get("id") == 8: load_roi = pts
                    elif item.get("id") == 7: unload_roi = pts
            print("ROI'ler yüklendi. (1:Machine, 8:Load, 7:Unload)")
        except Exception as e:
            print(f"rois.json okuma hatası: {e}")
    return machine_roi, load_roi, unload_roi

# --- ANA FONKSİYON ---
def main():
    machine_roi, load_roi, unload_roi = load_rois()

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Video açılamadı: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    frame_count = 0
    skip_counter = PLAYBACK_SKIP
    is_paused = False
    
    # ── İŞÇİ TAKİP ──
    # Her işçi: {state: "BOS"/"TASIYOR", start_frame, carried_part_id, last_bbox}
    worker_tracker = {}
    # Tamamlanan taşıma kayıtları
    worker_log = []

    # ── MAKİNE TAKİP (Hareket Tabanlı) ──
    # Her makine: {state: "IDLE"/"WORKING", start_frame, idle_start_frame, last_pos, current_dur, speed_history}
    machine_tracker = {}
    machine_log = []

    cv2.namedWindow("Istikbal Modern Analytics", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        if not is_paused:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            
            # Performans için frame atla (Hızlandırma)
            if frame_count % skip_counter != 0:
                continue
        else:
            # Duraklatılmışsa sadece klavyeyi dinle
            key = cv2.waitKey(30) & 0xFF
            if key == ord('p'): is_paused = False
            elif key == ord('q'): break
            continue

        # YOLO takip
        results = model.track(frame, persist=True, verbose=False)

        # Tespit edilen nesneleri sınıflarına ayır
        insanlar = []  # (id, bbox)
        makineler = [] # (id, bbox)
        parcalar = []  # (id, bbox)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)

            for bbox, obj_id, cls in zip(boxes, ids, clss):
                if cls == CLS_INSAN:
                    insanlar.append((obj_id, bbox))
                elif cls == CLS_MAKINE:
                    makineler.append((obj_id, bbox))
                elif cls == CLS_PARCA:
                    parcalar.append((obj_id, bbox))

        # ═══════════════════════════════════════
        # İŞÇİ + PARÇA TAŞIMA ANALİZİ
        # ═══════════════════════════════════════
        current_worker_ids = set()
        for w_id, w_bbox in insanlar:
            current_worker_ids.add(w_id)
            w_center = get_center(w_bbox)

            if w_id not in worker_tracker:
                worker_tracker[w_id] = {
                    "state": "BOS",
                    "start_frame": None,
                    "carried_part_id": None,
                    "last_bbox": w_bbox,
                    "no_part_counter": 0,
                    "last_carry_end_frame": 0,   # Son taşıma bitiş frame'i
                    "saved_start_frame": None     # Devam etmek için saklanan başlangıç
                }

            wt = worker_tracker[w_id]
            wt["last_bbox"] = w_bbox

            # İşçinin elinde parça var mı kontrol et
            touching_part = None
            best_overlap = 0.0
            for p_id, p_bbox in parcalar:
                if boxes_intersect(w_bbox, p_bbox):
                    iou = calc_iou(w_bbox, p_bbox)
                    if iou > best_overlap and iou >= MIN_OVERLAP_THRESHOLD:
                        best_overlap = iou
                        touching_part = p_id

            # Debug: Her 90 framede bir durumu bas
            if frame_count % 90 == 0:
                print(f"  [DEBUG] Isci {w_id}: state={wt['state']}, touching={touching_part}, overlap={best_overlap:.2f}")

            # --- STATE MACHINE ---
            if wt["state"] == "BOS":
                if touching_part is not None:
                    # Kısa süre önce taşıyordu ve tekrar parça aldı → sayacı DEVAM ETTİR
                    if (wt["saved_start_frame"] is not None and 
                        (frame_count - wt["last_carry_end_frame"]) < CARRY_COOLDOWN_FRAMES):
                        wt["state"] = "TASIYOR"
                        wt["start_frame"] = wt["saved_start_frame"]  # ESKİ sayaçtan devam
                        wt["carried_part_id"] = touching_part
                        wt["no_part_counter"] = 0
                        print(f"[İŞÇİ {w_id}] Parça tekrar algılandı, sayaç DEVAM EDİYOR (frame {frame_count})")
                    else:
                        # Yeni taşıma başlıyor
                        wt["state"] = "TASIYOR"
                        wt["start_frame"] = frame_count
                        wt["carried_part_id"] = touching_part
                        wt["no_part_counter"] = 0
                        wt["saved_start_frame"] = None
                        print(f"[İŞÇİ {w_id}] Parça aldı! (frame {frame_count}, overlap={best_overlap:.2f})")

            elif wt["state"] == "TASIYOR":
                # Bırakma alanında + parça artık elinde değil → KESİN TESLİM
                if is_inside(w_center, unload_roi) and touching_part is None:
                    duration = (frame_count - wt["start_frame"]) / fps
                    if duration >= MIN_CARRY_DURATION:
                        worker_log.append({"worker_id": w_id, "duration": duration, "frame": frame_count})
                        print(f"[İŞÇİ {w_id}] Parça teslim etti! Süre: {duration:.2f}s")
                    wt["state"] = "BOS"
                    wt["start_frame"] = None
                    wt["carried_part_id"] = None
                    wt["no_part_counter"] = 0
                    wt["saved_start_frame"] = None
                    wt["last_carry_end_frame"] = 0
                # Parça kayboldu → persistence bekle
                elif touching_part is None:
                    wt["no_part_counter"] += 1
                    if wt["no_part_counter"] > CARRY_PERSIST_FRAMES:
                        # Parça uzun süredir yok → geçici olarak BOS'a düş ama sayacı sakla
                        wt["state"] = "BOS"
                        wt["saved_start_frame"] = wt["start_frame"]  # Sayacı sakla!
                        wt["last_carry_end_frame"] = frame_count
                        wt["start_frame"] = None
                        wt["carried_part_id"] = None
                        wt["no_part_counter"] = 0
                else:
                    wt["no_part_counter"] = 0  # Parça hala elde

            # --- VİZÜALİZASYON ---
            if wt["state"] == "TASIYOR":
                color = (0, 0, 255)  # Kırmızı
                elapsed = (frame_count - wt["start_frame"]) / fps
                label = f"Isci {w_id}: PARCA TASINIYOR ({elapsed:.1f}s)"
            else:
                color = (200, 200, 200)  # Gri
                label = f"Isci {w_id}: BOS"

            cv2.rectangle(frame, (int(w_bbox[0]), int(w_bbox[1])),
                          (int(w_bbox[2]), int(w_bbox[3])), color, 2)
            cv2.putText(frame, label, (int(w_bbox[0]), int(w_bbox[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Kaybolmuş işçileri temizle
        lost_workers = [wid for wid in worker_tracker if wid not in current_worker_ids]
        for wid in lost_workers:
            if worker_tracker[wid]["state"] == "TASIYOR":
                dur = (frame_count - worker_tracker[wid]["start_frame"]) / fps
                if dur >= MIN_CARRY_DURATION:
                    worker_log.append({"worker_id": wid, "duration": dur, "frame": frame_count})
                    print(f"[İŞÇİ {wid}] Kayboldu (taşıma süresi: {dur:.2f}s)")
            del worker_tracker[wid]

        # ═══════════════════════════════════════
        # MAKİNE CYCLE TIME ANALİZİ
        # Yaklaşım: Frame-to-frame piksel farkı ile hareket algılama
        # ID değişse bile konum eşleşmesi ile tracker devam eder
        # ═══════════════════════════════════════
        
        # Mevcut makineleri tracker'larla eşleştir (konum bazlı)
        matched_tracker_keys = set()
        for m_id, m_bbox in makineler:
            m_center = get_center(m_bbox)
            
            # En yakın mevcut tracker'ı bul (ID veya konum ile)
            best_key = None
            if m_id in machine_tracker:
                best_key = m_id
            else:
                # ID eşleşmedi → konuma göre en yakın tracker'ı bul
                best_dist = 400  # Mesafe toleransı artırıldı (Flicker için)
                for tk, tv in machine_tracker.items():
                    if tk in matched_tracker_keys:
                        continue
                    d = np.sqrt((m_center[0]-tv["last_pos"][0])**2 + (m_center[1]-tv["last_pos"][1])**2)
                    if d < best_dist:
                        best_dist = d
                        best_key = tk
                        
                if best_key is not None and best_key != m_id:
                    # Eski tracker'ı yeni ID'ye taşı
                    machine_tracker[m_id] = machine_tracker.pop(best_key)
                    print(f"[MAKİNE] ID değişti: {best_key} → {m_id} (tracker devam ediyor)")
                    best_key = m_id
            
            # Yeni makine - tracker oluştur
            if best_key is None or m_id not in machine_tracker:
                machine_tracker[m_id] = {
                    "state": "IDLE",
                    "start_frame": None,
                    "idle_start_frame": None,
                    "last_pos": m_center,
                    "current_dur": 0.0,
                    "prev_crop": None,
                    "home_crop": None,              # Cycle başındaki referans görüntü
                    "has_moved_away": False,        # Makine evden uzaklaştı mı?
                    "max_h_diff": 0.0,              # Cycle içindeki max fark
                    "last_seen_frame": frame_count,
                    "last_bbox": m_bbox,
                }
            
            mt = machine_tracker[m_id]
            mt["last_seen_frame"] = frame_count
            mt["last_pos"] = m_center
            mt["last_bbox"] = m_bbox
            matched_tracker_keys.add(m_id)
            
            # --- Makine bbox'ının üst kısmını crop et (boru bölgesi) ---
            x1, y1, x2, y2 = int(m_bbox[0]), int(m_bbox[1]), int(m_bbox[2]), int(m_bbox[3])
            h = y2 - y1
            pipe_y2 = y1 + int(h * 0.3) # Sadece üst %30 (Borunun olduğu yer)
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(frame.shape[1], x2)
            pipe_y2 = min(frame.shape[0], max(pipe_y2, y1 + 1))
            
            crop = frame[y1:pipe_y2, x1:x2]
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size > 0 else None
            
            # --- Frame-to-frame piksel farkı (Anlık Hareket) ---
            pixel_diff = 0.0
            if mt["prev_crop"] is not None and crop_gray is not None:
                try:
                    if mt["prev_crop"].shape == crop_gray.shape:
                        diff = cv2.absdiff(crop_gray, mt["prev_crop"])
                        pixel_diff = np.mean(diff)
                    else:
                        prev_resized = cv2.resize(mt["prev_crop"], (crop_gray.shape[1], crop_gray.shape[0]))
                        diff = cv2.absdiff(crop_gray, prev_resized)
                        pixel_diff = np.mean(diff)
                except:
                    pixel_diff = 0.0
            is_moving = pixel_diff > MACHINE_MOTION_THRESHOLD
            
            # --- Referans (Home) görüntüsüne olan fark ---
            home_diff = 100.0 # Varsayılan yüksek fark
            if mt["home_crop"] is not None and crop_gray is not None:
                try:
                    if mt["home_crop"].shape == crop_gray.shape:
                        h_diff = cv2.absdiff(crop_gray, mt["home_crop"])
                        home_diff = np.mean(h_diff)
                    else:
                        home_resized = cv2.resize(mt["home_crop"], (crop_gray.shape[1], crop_gray.shape[0]))
                        h_diff = cv2.absdiff(crop_gray, home_resized)
                        home_diff = np.mean(h_diff)
                except:
                    home_diff = 100.0

            mt["prev_crop"] = crop_gray
            
            # Hareket yoksa Home görüntüsünü güncelle (En güncel durma hali)
            if not is_moving and crop_gray is not None:
                mt["home_crop"] = crop_gray
            
            # Makine evden uzaklaştı mı?
            if mt["state"] == "WORKING":
                if home_diff > mt["max_h_diff"]:
                    mt["max_h_diff"] = home_diff
                
                if home_diff > 5.0: # En az 5 piksel fark oluşmalı
                    mt["has_moved_away"] = True

            # Debug
            if frame_count % 60 == 0:
                print(f"  [M-DEBUG] Makine {m_id}: state={mt['state']}, p_diff={pixel_diff:.1f}, h_diff={home_diff:.1f}, max_h={mt['max_h_diff']:.1f}, away={mt['has_moved_away']}")
            
            # --- STATE MACHINE ---
            if mt["state"] == "IDLE":
                if is_moving:
                    mt["state"] = "WORKING"
                    mt["start_frame"] = frame_count
                    mt["idle_start_frame"] = None
                    mt["has_moved_away"] = False
                    mt["max_h_diff"] = 0.0
                    print(f"[MAKİNE {m_id}] Cycle başladı!")
            
            elif mt["state"] == "WORKING":
                cycle_ended = False
                
                # Sadece Durma timeout (Süre başa sarmasın diye 'eve döndü' kontrolünü kaldırdık)
                if not is_moving:
                    if mt["idle_start_frame"] is None:
                        mt["idle_start_frame"] = frame_count
                    else:
                        idle_duration = (frame_count - mt["idle_start_frame"]) / fps
                        if idle_duration >= MACHINE_IDLE_TIMEOUT:
                            duration = (mt["idle_start_frame"] - mt["start_frame"]) / fps
                            if duration >= MIN_CYCLE_DURATION:
                                cycle_ended = True
                                print(f"[MAKİNE {m_id}] İşlem tamamlandı (durdu), süre: {duration:.2f}s")
                else:
                    mt["idle_start_frame"] = None

                if cycle_ended:
                    duration = (frame_count - mt["start_frame"]) / fps
                    machine_log.append({
                        "machine_id": m_id,
                        "cycle_time": duration,
                        "frame": frame_count
                    })
                    print(f"[MAKİNE {m_id}] Cycle kaydedildi: {duration:.2f}s")
                    mt["state"] = "IDLE"
                    mt["start_frame"] = None
                    mt["idle_start_frame"] = None
                    mt["has_moved_away"] = False
                    mt["max_h_diff"] = 0.0
            
            # Anlık süre
            if mt["state"] == "WORKING" and mt["start_frame"] is not None:
                mt["current_dur"] = (frame_count - mt["start_frame"]) / fps
            else:
                mt["current_dur"] = 0.0
            
            # --- VİZÜALİZASYON (Makine) ---
            if mt["state"] == "WORKING":
                m_color = (0, 255, 100) if is_moving else (0, 180, 255)
                m_label = f"M{m_id} ACTIVE"
            else:
                m_color = (180, 180, 180)
                m_label = f"M{m_id} IDLE"
            
            # Zarif Köşeli Box
            x1, y1, x2, y2 = int(m_bbox[0]), int(m_bbox[1]), int(m_bbox[2]), int(m_bbox[3])
            cv2.rectangle(frame, (x1, y1), (x2, y2), m_color, 1, cv2.LINE_AA)
            frame = draw_styled_text(frame, m_label, (x1, y1-25), 16, m_color[::-1], True)

            # Modern Status Barları
            bar_x, bar_y = x1, y2 + 8
            # P_DIFF (Hız)
            p_w = min(int(pixel_diff * 10), 100)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 100, bar_y + 4), (50, 50, 50), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + p_w, bar_y + 4), (255, 150, 0), -1)
            # H_DIFF (Tur İlerlemesi)
            h_w = min(int(home_diff * 2), 100)
            cv2.rectangle(frame, (bar_x, bar_y + 8), (bar_x + 100, bar_y + 12), (50, 50, 50), -1)
            cv2.rectangle(frame, (bar_x, bar_y + 8), (bar_x + h_w, bar_y + 12), (0, 255, 255), -1)
        
        # Kaybolmuş makineleri temizle
        lost_machines = [mid for mid in machine_tracker if mid not in matched_tracker_keys]
        for mid in list(lost_machines):
            mt = machine_tracker[mid]
            age = frame_count - mt["last_seen_frame"]
            
            if mt["state"] == "WORKING" and mt["start_frame"] is not None:
                mt["current_dur"] = (frame_count - mt["start_frame"]) / fps
                m_bbox = mt["last_bbox"]
                cv2.rectangle(frame, (int(m_bbox[0]), int(m_bbox[1])), (int(m_bbox[2]), int(m_bbox[3])), (100, 100, 100), 1, cv2.LINE_8)
                frame = draw_styled_text(frame, f"M{mid} TRACING", (int(m_bbox[0]), int(m_bbox[1])-20), 14, (150, 150, 150), False)

            if age > MACHINE_PERSIST_FRAMES:
                if mt["state"] == "WORKING" and mt["start_frame"] is not None:
                    cycle_time = (frame_count - mt["start_frame"]) / fps
                    if cycle_time >= MIN_CYCLE_DURATION:
                        machine_log.append({"machine_id": mid, "cycle_time": cycle_time, "frame": frame_count})
                del machine_tracker[mid]
            # else: tracker korunuyor, makine geri gelecek

        # ═══════════════════════════════════════
        # PARÇA VİZÜALİZASYONU
        # ═══════════════════════════════════════
        for p_id, p_bbox in parcalar:
            cv2.rectangle(frame, (int(p_bbox[0]), int(p_bbox[1])),
                          (int(p_bbox[2]), int(p_bbox[3])), (0, 255, 255), 2)
            cv2.putText(frame, f"Parca {p_id}", (int(p_bbox[0]), int(p_bbox[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # ═══════════════════════════════════════
        # ROI ÇİZİMİ
        # ═══════════════════════════════════════
        if load_roi is not None and load_roi.size > 0:
            cv2.polylines(frame, [load_roi], True, (0, 255, 255), 3)
            cv2.putText(frame, "YUKLEME ALANI", (load_roi[0][0], load_roi[0][1]-10), 1, 1.5, (0, 255, 255), 2)
        if unload_roi is not None and unload_roi.size > 0:
            cv2.polylines(frame, [unload_roi], True, (255, 0, 255), 3)
            cv2.putText(frame, "BIRAKMA ALANI", (unload_roi[0][0], unload_roi[0][1]-10), 1, 1.5, (255, 0, 255), 2)
        if machine_roi is not None and machine_roi.size > 0:
            cv2.polylines(frame, [machine_roi], True, (0, 255, 0), 3)
            cv2.putText(frame, "MAKINE HOME", (machine_roi[0][0], machine_roi[0][1]-10), 1, 1.5, (0, 255, 0), 2)

        # ═══════════════════════════════════════
        # MODERN PREMIUM DASHBOARD
        # ═══════════════════════════════════════
        frame = draw_glass_panel(frame, 20, 20, 420, 500, (40, 30, 20), 0.75)
        
        # Başlık
        frame = draw_styled_text(frame, "İSTİKBAL", (40, 40), 28, (255, 255, 255), True)
        frame = draw_styled_text(frame, "INDUSTRIAL ANALYTICS", (40, 75), 14, (200, 200, 200), False)
        cv2.line(frame, (40, 100), (390, 100), (255, 255, 255), 1, cv2.LINE_AA)

        y = 120
        # --- İŞÇİ PANELİ ---
        frame = draw_styled_text(frame, "HUMAN ACTIVITY", (40, y), 18, (0, 200, 255), True)
        y += 35
        for wid, wt in worker_tracker.items():
            color = (0, 255, 100) if wt["state"] == "TASIYOR" else (200, 200, 200)
            status = f"WORKING {((frame_count - wt['start_frame'])/fps):.1f}s" if wt["state"] == "TASIYOR" else "IDLE"
            frame = draw_styled_text(frame, f"Worker {wid}: {status}", (50, y), 15, color[::-1], False)
            y += 25
        
        y += 20
        # --- MAKİNE PANELİ ---
        frame = draw_styled_text(frame, "MACHINE CYCLES", (40, y), 18, (100, 255, 0), True)
        y += 35
        for mid, mt in machine_tracker.items():
            color = (0, 255, 100) if mt["state"] == "WORKING" else (180, 180, 180)
            status = f"RUNNING {mt['current_dur']:.1f}s" if mt["state"] == "WORKING" else "IDLE"
            frame = draw_styled_text(frame, f"Machine {mid}: {status}", (50, y), 15, color[::-1], False)
            y += 25

        y += 30
        # --- ÖZET İSTATİSTİKLER ---
        cv2.rectangle(frame, (40, y), (390, y+80), (60, 50, 40), -1)
        frame = draw_styled_text(frame, f"Total Cycles: {len(machine_log)}", (60, y+15), 16, (255, 255, 255), True)
        frame = draw_styled_text(frame, f"Total Carries: {len(worker_log)}", (60, y+45), 16, (255, 255, 255), True)

        # Alt Bilgi
        frame = draw_styled_text(frame, f"FPS: {fps:.1f} | Frame: {frame_count}", (40, 470), 12, (150, 150, 150), False)

        # Göster
        cv2.imshow("Istikbal Modern Analytics", frame)
        
        # --- KLAVYE KONTROLLERİ ---
        key = cv2.waitKey(PLAYBACK_DELAY) & 0xFF
        if key == ord('q'): 
            break
        elif key == ord('p'): 
            is_paused = not is_paused
        elif key == ord('f'): # Daha Hızlı
            skip_counter = min(skip_counter + 1, 10)
            print(f"[HIZ] Frame Skip: {skip_counter}")
        elif key == ord('s'): # Daha Yavaş
            skip_counter = max(skip_counter - 1, 1)
            print(f"[HIZ] Frame Skip: {skip_counter}")

    cap.release()
    cv2.destroyAllWindows()

    # Final rapor
    print("\n" + "="*50)
    print("FINAL RAPOR")
    print("="*50)
    print(f"\nToplam İşçi Taşıma: {len(worker_log)}")
    for w in worker_log:
        print(f"  İşçi {w['worker_id']}: {w['duration']:.2f}s (frame {w['frame']})")
    print(f"\nToplam Makine Cycle: {len(machine_log)}")
    for m in machine_log:
        print(f"  Makine {m['machine_id']}: {m['cycle_time']:.2f}s (frame {m['frame']})")

if __name__ == "__main__":
    main()
