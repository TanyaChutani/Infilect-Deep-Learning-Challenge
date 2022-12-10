cd /content/Retail-Store-Item-Detection-using-YOLOv5
python detect.py --weights /content/Retail-Store-Item-Detection-using-YOLOv5/last_yolov5s_results.pt --img 416 --conf 0.4 --source /content/destination/product_detection_from_packshots/shelf_images --save-txt
