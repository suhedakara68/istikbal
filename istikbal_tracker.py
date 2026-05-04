import cv2
import numpy as np
from ultralytics import YOLO
import json
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- CONFIG ---
MODEL_PATH = "best.pt"
VIDEO_PATH = "istikball.mp4"

# Sınıf ID'leri (yeni model)
CLS_INSAN = 0
CLS_MAKINE = 1
CLS_PARCA = 2

# Eşikler  
MACHINE_IDLE_TIMEOUT = 3.0     # Makine durup bu kadar sn beklerse cycle biter
MACHINE_MOTION_THRESHOLD = 1.0 # Piksel farkı bu değerin üstündeyse hareket var
MACHINE_PERSIST_FRAMES = 150   # Makine kaybolsa bile bu kadar frame tracker korunur (5sn@30fps)
MIN_CYCLE_DURATION = 2.0       # Bu süreden kısa cycle'lar sayılmaz
MIN_CARRY_DURATION = 0.5       # Çok kısa taşıma süreleri filtrelenir
BOX_INTERSECT_TOLERANCE = 50   # İşçi-parça çakışma toleransı (px)
MIN_OVERLAP_THRESHOLD = 0.30   # Minimum overlap oranı (parça taşıyor sayılması için)
CARRY_PERSIST_FRAMES = 60      # Parça kaybedilse bile 2sn boyunca taşıma devam eder
CARRY_COOLDOWN_FRAMES = 90     # Bu süre içinde tekrar parça alırsa sayaç devam eder (sıfırlanmaz)

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
    print(f"Video açıldı: {VIDEO_PATH} | FPS: {fps:.1f}")

    frame_count = 0

    # ── İŞÇİ TAKİP ──
    # Her işçi: {state: "BOS"/"TASIYOR", start_frame, carried_part_id, last_bbox}
    worker_tracker = {}
    # Tamamlanan taşıma kayıtları
    worker_log = []

    # ── MAKİNE TAKİP (Hareket Tabanlı) ──
    # Her makine: {state: "IDLE"/"WORKING", start_frame, idle_start_frame, last_pos, current_dur, speed_history}
    machine_tracker = {}
    machine_log = []

    cv2.namedWindow("Istikbal Tracker", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

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
                best_dist = 200  # Max 200px uzaklık
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
                    "last_seen_frame": frame_count,
                    "last_bbox": m_bbox,
                }
            
            mt = machine_tracker[m_id]
            mt["last_seen_frame"] = frame_count
            mt["last_pos"] = m_center
            mt["last_bbox"] = m_bbox
            matched_tracker_keys.add(m_id)
            
            # --- Makine bbox'ının üst yarısını crop et (boru bölgesi) ---
            x1, y1, x2, y2 = int(m_bbox[0]), int(m_bbox[1]), int(m_bbox[2]), int(m_bbox[3])
            h = y2 - y1
            pipe_y2 = y1 + int(h * 0.5)
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(frame.shape[1], x2)
            pipe_y2 = min(frame.shape[0], max(pipe_y2, y1 + 1))
            
            crop = frame[y1:pipe_y2, x1:x2]
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size > 0 else None
            
            # --- Frame-to-frame piksel farkı ---
            pixel_diff = 0.0
            if mt["prev_crop"] is not None and crop_gray is not None:
                try:
                    if mt["prev_crop"].shape == crop_gray.shape:
                        diff = cv2.absdiff(crop_gray, mt["prev_crop"])
                    else:
                        prev_resized = cv2.resize(mt["prev_crop"], (crop_gray.shape[1], crop_gray.shape[0]))
                        diff = cv2.absdiff(crop_gray, prev_resized)
                    pixel_diff = np.mean(diff)
                except:
                    pixel_diff = 0.0
            
            mt["prev_crop"] = crop_gray
            
            # Hareket var mı? (basit threshold)
            is_moving = pixel_diff > MACHINE_MOTION_THRESHOLD
            
            # Debug
            if frame_count % 60 == 0:
                print(f"  [M-DEBUG] Makine {m_id}: state={mt['state']}, pixel_diff={pixel_diff:.2f}, moving={is_moving}, dur={mt['current_dur']:.1f}s")
            
            # --- STATE MACHINE ---
            if mt["state"] == "IDLE":
                if is_moving:
                    mt["state"] = "WORKING"
                    mt["start_frame"] = frame_count
                    mt["idle_start_frame"] = None
                    print(f"[MAKİNE {m_id}] Cycle başladı! (frame {frame_count})")
            
            elif mt["state"] == "WORKING":
                if is_moving:
                    # Hareket var → idle timer sıfırla, cycle devam
                    mt["idle_start_frame"] = None
                else:
                    # Hareket yok → idle timer başlat
                    if mt["idle_start_frame"] is None:
                        mt["idle_start_frame"] = frame_count
                    else:
                        idle_duration = (frame_count - mt["idle_start_frame"]) / fps
                        if idle_duration >= MACHINE_IDLE_TIMEOUT:
                            # Cycle bitti!
                            cycle_time = (mt["idle_start_frame"] - mt["start_frame"]) / fps
                            if cycle_time >= MIN_CYCLE_DURATION:
                                machine_log.append({
                                    "machine_id": m_id,
                                    "cycle_time": cycle_time,
                                    "frame": frame_count
                                })
                                print(f"[MAKİNE {m_id}] Cycle bitti! Süre: {cycle_time:.2f}s (frame {frame_count})")
                            mt["state"] = "IDLE"
                            mt["start_frame"] = None
                            mt["idle_start_frame"] = None
            
            # Anlık süre
            if mt["state"] == "WORKING" and mt["start_frame"] is not None:
                mt["current_dur"] = (frame_count - mt["start_frame"]) / fps
            else:
                mt["current_dur"] = 0.0
            
            # --- VİZÜALİZASYON ---
            if mt["state"] == "WORKING":
                if is_moving:
                    m_color = (0, 255, 0)   # Yeşil
                    m_label = f"Makine {m_id}: CALISIYOR ({mt['current_dur']:.1f}s)"
                else:
                    m_color = (0, 200, 255)  # Turuncu - kısa duraklama
                    m_label = f"Makine {m_id}: CALISIYOR ({mt['current_dur']:.1f}s)"
            else:
                m_color = (150, 150, 150)
                m_label = f"Makine {m_id}: IDLE"
            
            cv2.rectangle(frame, (int(m_bbox[0]), int(m_bbox[1])),
                          (int(m_bbox[2]), int(m_bbox[3])), m_color, 2)
            cv2.putText(frame, m_label, (int(m_bbox[0]), int(m_bbox[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, m_color, 2)
            
            # Boru bölgesi + hareket barı
            cv2.rectangle(frame, (x1, y1), (x2, pipe_y2), (255, 255, 0), 1)
            bar_x = int(m_bbox[0])
            bar_y = int(m_bbox[3]) + 5
            bar_len = min(int(pixel_diff * 10), 200)
            bar_color = (0, 255, 0) if is_moving else (0, 0, 255)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_len, bar_y + 8), bar_color, -1)
            cv2.putText(frame, f"diff:{pixel_diff:.1f}", (bar_x, bar_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Kaybolmuş makineleri temizle (persistence ile - hemen silme!)
        lost_machines = [mid for mid in machine_tracker if mid not in matched_tracker_keys]
        for mid in list(lost_machines):
            age = frame_count - machine_tracker[mid]["last_seen_frame"]
            if age > MACHINE_PERSIST_FRAMES:
                # Çok uzun süredir kayıp → sil
                if machine_tracker[mid]["state"] == "WORKING" and machine_tracker[mid]["start_frame"] is not None:
                    cycle_time = (frame_count - machine_tracker[mid]["start_frame"]) / fps
                    if cycle_time >= MIN_CYCLE_DURATION:
                        machine_log.append({"machine_id": mid, "cycle_time": cycle_time, "frame": frame_count})
                        print(f"[MAKİNE {mid}] Kayboldu (cycle süresi: {cycle_time:.2f}s)")
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
        # DASHBOARD (SOL ÜST KÖŞE)
        # ═══════════════════════════════════════
        overlay = frame.copy()
        panel_h = 60 + len(worker_log[-3:]) * 30 + len(machine_log[-3:]) * 30 + 120
        cv2.rectangle(overlay, (10, 10), (500, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        y = 35
        cv2.putText(frame, "=== ISCI SAYACI ===", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += 30

        # Aktif işçiler
        for wid, wt in worker_tracker.items():
            if wt["state"] == "TASIYOR":
                elapsed = (frame_count - wt["start_frame"]) / fps
                cv2.putText(frame, f"  Isci {wid}: TASIYOR {elapsed:.1f}s", (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"  Isci {wid}: BOS", (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y += 25

        # Son tamamlanan taşımalar
        if worker_log:
            last_w = worker_log[-1]
            cv2.putText(frame, f"  Son Tasima: Isci {last_w['worker_id']} = {last_w['duration']:.2f}s", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
            y += 25
        cv2.putText(frame, f"  Toplam Tasima: {len(worker_log)}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 35

        cv2.putText(frame, "=== MAKINE SAYACI ===", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 30

        # Aktif makineler
        for mid, mt in machine_tracker.items():
            if mt["state"] != "IDLE":
                cv2.putText(frame, f"  Makine {mid}: {mt['state']} {mt['current_dur']:.1f}s", (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"  Makine {mid}: IDLE", (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            y += 25

        # Son cycle
        if machine_log:
            last_m = machine_log[-1]
            cv2.putText(frame, f"  Son Cycle: Makine {last_m['machine_id']} = {last_m['cycle_time']:.2f}s", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
            y += 25
        cv2.putText(frame, f"  Toplam Cycle: {len(machine_log)}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Göster
        cv2.imshow("Istikbal Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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
