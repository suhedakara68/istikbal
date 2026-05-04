import cv2
import json
import numpy as np
import os

drawing = False
current_pts = []
rois = []
frame = None
display_frame = None
dragging_point = None
delete_mode = False

def mouse_callback(event, x, y, flags, param):
    global current_pts, frame, display_frame, rois, dragging_point, delete_mode
    
    if delete_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, r in enumerate(rois):
                if "pts" in r:
                    pts_arr = np.array(r["pts"], np.int32)
                    if cv2.pointPolygonTest(pts_arr, (x, y), False) >= 0:
                        print(f"Silinen ROI: {r['name']}")
                        rois.pop(i)
                        update_display()
                        break
        return
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Tıklanan yer mevcut bir noktaya yakın mı? Sürükleme başlatalım
        for i, r in enumerate(rois):
            if "pts" in r:
                for j, p in enumerate(r["pts"]):
                    if (p[0]-x)**2 + (p[1]-y)**2 < 100: # 10 piksel yarıçap
                        dragging_point = (i, j)
                        return
                        
        # Yeni nokta ekle
        current_pts.append([x, y])
        update_display()
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_point is not None:
            i, j = dragging_point
            rois[i]["pts"][j] = [x, y]
            update_display()
            
    elif event == cv2.EVENT_LBUTTONUP:
        if dragging_point is not None:
            dragging_point = None
            update_display()
            
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current_pts) > 0:
            current_pts.pop()
            update_display()
        else:
            # Çizim modunda değilsek ve bir ROI'nin içine sağ tıkladıysak onu sil
            for i, r in enumerate(rois):
                if "pts" in r:
                    pts_arr = np.array(r["pts"], np.int32)
                    if cv2.pointPolygonTest(pts_arr, (x, y), False) >= 0:
                        print(f"Silinen ROI: {r['name']}")
                        rois.pop(i)
                        update_display()
                        break

def update_display():
    global frame, display_frame, current_pts, rois
    display_frame = frame.copy()
    
    # Çizilmiş olanları göster
    for r in rois:
        if "pts" in r:
            pts_array = []
            for p in r["pts"]:
                pts_array.append(p)
            cv2.polylines(display_frame, [np.array(pts_array, np.int32)], True, (0, 0, 255), 2)
            cv2.putText(display_frame, r["name"], (pts_array[0][0], pts_array[0][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
    # Şu an çizilmekte olanı göster
    if len(current_pts) > 0:
        for i in range(len(current_pts)):
            cv2.circle(display_frame, current_pts[i], 4, (0, 255, 0), -1)
            if i > 0:
                cv2.line(display_frame, current_pts[i-1], current_pts[i], (0, 255, 0), 2)
        
        # Olası kapatma çizgisi (ilk noktaya dönüş)
        cv2.line(display_frame, current_pts[-1], current_pts[0], (255, 255, 0), 1)
        
    if delete_mode:
        cv2.putText(display_frame, "SILME MODU AKTIF (Kapatmak icin 2'ye basin)", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
    cv2.imshow("ROI Olusturucu", display_frame)


def main():
    global frame, display_frame, current_pts, rois, delete_mode
    
    source = 'istikball.mp4'
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Hata: {source} acilamadi. Dosya yolunu kontrol edin.")
        return
        
    success, frame = cap.read()
    cap.release()
    if not success:
        print("Hata: Videodan kare okunamadi.")
        return
        
    ref_h, ref_w = frame.shape[:2]
    
    roi_id_counter = 0
    # Mevcut rois.json varsa yükle
    if os.path.exists("rois.json"):
        try:
            with open("rois.json", "r", encoding="utf-8") as f:
                loaded_rois = json.load(f)
                for r in loaded_rois:
                    rois.append(r)
                    if "id" in r and isinstance(r["id"], int) and r["id"] >= roi_id_counter:
                        roi_id_counter = r["id"] + 1
            print(f"Mevcut {len(rois)} adet ROI yüklendi.")
        except Exception as e:
            print(f"rois.json okunamadı: {e}")

    cv2.namedWindow("ROI Olusturucu", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("ROI Olusturucu", mouse_callback)
    
    # Ekranda talimatlar
    intructions = [
        "Sol Tık (Boş): Nokta ekle",
        "Sol Tık (Nokta Üzerinde): Noktayı sürükle/düzenle",
        "Sağ Tık (Çizimde): Son noktayı sil",
        "Sağ Tık (ROI İçinde): ROI'yi sil",
        "2: Silme Modunu Aç/Kapat (Açıkken ROI'lere sol tık ile silinir)",
        "1/Enter/C: Mevcut çokgeni tamamla ve listeye ekle",
        "S: Tüm ROI'leri rois.json olarak dışarı aktar ve Çık",
        "Q/Esc: Kaydetmeden çık"
    ]
    
    print("ROI Oluşturucu Başlatıldı.")
    for ins in intructions:
        print("-", ins)

    # Baslangic cizimi
    update_display()
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27: # q veya esc
            print("Kaydetmeden çıkılıyor...")
            break
            
        elif key == ord('2'):
            delete_mode = not delete_mode
            print("Silme Modu:", "AKTİF" if delete_mode else "KAPALI")
            update_display()
            
        elif key == 13 or key == ord('c') or key == ord('1'): # Enter, c veya 1 tusu
            if len(current_pts) >= 3:
                name = f"id={roi_id_counter}"
                print(f"{name} eklendi: {current_pts}")
                rois.append({
                    "id": roi_id_counter,
                    "pts": list(current_pts),
                    "name": name,
                    "ref_w": ref_w,
                    "ref_h": ref_h
                })
                roi_id_counter += 1
                current_pts = []
                update_display()
            else:
                print("Hata: Bir çokgen oluşturmak için en az 3 nokta seçmelisiniz!")
                
        elif key == ord('s'):
            if len(rois) > 0:
                with open("rois.json", "w", encoding="utf-8") as f:
                    json.dump(rois, f, indent=4)
                print(f"Başarılı: {len(rois)} adet ROI rois.json dosyasına kaydedildi.")
                print("detect.py artık bu rois.json'i otomatik okuyacaktır.")
            else:
                print("Uyarı: Kaydedilecek hiç ROI yok.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
